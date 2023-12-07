# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : train.py
@Function: XX
@Other: XX
"""
import datetime
import os
import shutil
import logging
import torch
from utils.functions import set_seed, set_logger, save_json, reset_console
from utils.train_models import TrainLayerLabeling
import config
import json
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pickle

args = config.Args().get_parser()
logger = logging.getLogger(__name__)


class LayerDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [torch.as_tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.as_tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.as_tensor(example.token_type_ids, dtype=torch.long) for example in features]
        self.masks = [torch.as_tensor(example.masks, dtype=torch.uint8) for example in features]
        self.labels = [torch.as_tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'masks': self.masks[index],
                'labels': self.labels[index]}

        return data


if __name__ == '__main__':
    args.data_name = os.path.basename(os.path.abspath(args.data_dir))
    args.model_name = os.path.basename(os.path.abspath(args.bert_dir))
    args.save_path = os.path.join('./checkpoints',
                                  args.data_name + '-' + args.model_name
                                  + '-' + str(datetime.date.today()))

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    # 复制对应的labels文件
    shutil.copy(os.path.join(args.data_dir, 'labels.json'), os.path.join(args.save_path, 'labels.json'))

    set_logger(os.path.join(args.save_path, 'log.txt'))
    torch.set_float32_matmul_precision('high')

    if args.data_name == "财务附注定位":
        # set_seed(args.seed)
        args.task_type = 'layering'
        args.max_seq_len = 128  # 滑窗视野为128
        args.max_word_len = 40  # 每行最多取40个token
        args.use_crf = False

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        reset_console(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = LayerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = LayerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

    if args.task_type == 'layering':
        myModel = TrainLayerLabeling(args, train_loader, dev_loader, label_list, logger)
        myModel.train()
