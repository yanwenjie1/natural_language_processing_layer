# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:04
@File    : train_models.py
@Function: XX
@Other: XX
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pynvml
from tqdm import tqdm
from utils.models import LayerLabeling
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model, get_entity_bieos
from utils.adversarial_training import PGD


class TrainLayerLabeling:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.log = log
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.device = load_model_and_parallel(LayerLabeling(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)

    def loss(self, model_output, true_label, attention_masks):
        if self.args.use_crf:
            loss = -self.model.crf(model_output, true_label, mask=attention_masks.bool(), reduction='mean')
        else:
            active_loss = attention_masks.view(-1) == 1
            active_logits = model_output.view(-1, model_output.size()[2])[active_loss]
            active_labels = true_label.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        return loss

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(1, self.args.train_epochs + 1):  # 训练epoch数 默认50
            bar = tqdm(self.train_loader, ncols=80)
            losses = []
            for batch_data in bar:
                self.model.train()
                # model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
                # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss = self.loss(train_outputs, batch_data['labels'], batch_data['masks'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                bar.set_description("[epoch] %s" % str(epoch))
                # loss.backward(loss.clone().detach())
                loss.backward()  # 反向传播 计算当前梯度
                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)(更新了n次embedding层)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 在对抗的最后一次恢复一开始保存的梯度 这时候的embedding参数层也加了3次扰动!
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss_adv = self.loss(train_outputs_adv, batch_data['labels'], batch_data['masks'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                        bar.set_description("[epoch] %s" % str(epoch))
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数层

                # 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
                # optimizer.zero_grad()的作用是清除优化器涉及的所有torch.Tensor的梯度 当模型只用了一个优化器时 是等价的
            if epoch > self.args.train_epochs * 0.3:
                dev_loss, precision, recall, f1 = self.dev()
                if f1 > best_f1:
                    best_f1 = f1
                    save_model(self.args, self.model, str(epoch) + '_{:.4f}'.format(f1), self.log)
                self.log.info('[eval] epoch:{} loss:{:.6f} precision={:.6f} recall={:.6f} f1={:.6f} best_f1={:.6f}'
                              .format(epoch, dev_loss, precision, recall, f1, best_f1))
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.log.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in tqdm(self.dev_loader, leave=False, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_outputs = self.model(dev_batch_data['token_ids'],
                                         dev_batch_data['attention_masks'],
                                         dev_batch_data['token_type_ids'],
                                         'dev')
                dev_loss = self.loss(dev_outputs, dev_batch_data['labels'], dev_batch_data['masks'])
                tot_dev_loss += dev_loss.detach().item()
                if self.args.use_crf:
                    batch_output = self.model.crf.decode(dev_outputs, mask=dev_batch_data['masks'])
                else:
                    batch_output = dev_outputs.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=-1)
                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity_bieos([self.id2label[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity_bieos([self.id2label[i] for i in y_true_list]))
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return tot_dev_loss, precision, recall, f1

    def test(self, model_path):
        model, device = load_model_and_parallel(LayerLabeling(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 根据label确定有哪些实体类
        tags = [item[1] for item in self.id2label.items()]
        tags.remove('O')
        tags = [v[2:] for v in tags]
        entitys = list(set(tags))
        entitys.sort()
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                dev_outputs = model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    'dev')
                if self.args.use_crf:
                    batch_output = self.model.crf.decode(dev_outputs, mask=dev_batch_data['masks'])
                else:
                    batch_output = dev_outputs.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=-1)

                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity_bieos([self.id2label[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity_bieos([self.id2label[i] for i in y_true_list]))
                    X_all += len(R & T)
                    Y_all += len(R)
                    Z_all += len(T)
                    for item in R & T:
                        X[entitys_to_ids[item[0]]] += 1
                    for item in R:
                        Y[entitys_to_ids[item[0]]] += 1
                    for item in T:
                        Z[entitys_to_ids[item[0]]] += 1

        # len1 = max(max([len(i) for i in entitys]), 4)
        # f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all
        # str_log = '\n{:<10}{:<15}{:<15}{:<15}\n'.format('实体' + chr(12288) * (len1 - len('实体')), 'precision', 'recall',
        #                                                 'f1-score')
        # str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format('全部实体' + chr(12288) * (len1 - len('全部实体')), precision,
        #                                                         recall, f1)
        # # logger.info('all_entity: precision:{:.6f}, recall:{:.6f}, f1-score:{:.6f}'
        # #             .format(precision, recall, f1))
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        # for entity in entitys:
        #     str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format(entity + chr(12288) * (len1 - len(entity)),
        #                                                             precision[entitys_to_ids[entity]],
        #                                                             recall[entitys_to_ids[entity]],
        #                                                             f1[entitys_to_ids[entity]])

        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all

        str_log = '\n' + '实体\t' + 'precision\t' + 'pre_count\t' + 'recall\t' + 'true_count\t' + 'f1-score\n'
        str_log += '' \
                   + '全部实体\t' \
                   + '%.4f' % precision + '\t' \
                   + '%.0f' % Y_all + '\t' \
                   + '%.4f' % recall + '\t' \
                   + '%.0f' % Z_all + '\t' \
                   + '%.4f' % f1 + '\n'
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        for entity in entitys:
            str_log += '' \
                       + entity + '\t' \
                       + '%.4f' % precision[entitys_to_ids[entity]] + '\t' \
                       + '%.0f' % Y[entitys_to_ids[entity]] + '\t' \
                       + '%.4f' % recall[entitys_to_ids[entity]] + '\t' \
                       + '%.0f' % Z[entitys_to_ids[entity]] + '\t' \
                       + '%.4f' % f1[entitys_to_ids[entity]] + '\n'
        self.log.info(str_log)
