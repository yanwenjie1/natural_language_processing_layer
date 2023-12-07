# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/5/9
@Time    : 13:12
@File    : server_confindence.py
@Function: XX
@Other: XX
"""

import copy
import json
import os
import torch
import socket
import numpy as np
import tqdm
from flask import Flask, request
from gevent import pywsgi
from transformers import BertTokenizer
from utils.functions import load_model_and_parallel, get_entity_bieos
from utils.models import LayerLabeling


def torch_env():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    """
    ip获取
    :return:
    """
    myIp = [item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None) if ':' not in item[4][0]][0]
    return myIp


def encode(texts):
    """

    :param texts: list of str
    :return:
    """
    assert type(texts) == list
    results = []

    # 按照 max_seq_len 拼接语料
    start = 0
    while start < len(texts):
        texts_part = texts[start: start + args.max_seq_len]

        word_ids = tokenizer.batch_encode_plus(texts_part,
                                               max_length=args.max_word_len,
                                               padding="max_length",
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        token_ids = word_ids['input_ids'].unsqueeze(0)
        attention_masks = word_ids['attention_mask'].unsqueeze(0)
        token_type_ids = word_ids['token_type_ids'].unsqueeze(0)
        masks = torch.as_tensor(np.ones((1, len(texts_part))), dtype=torch.bool)

        # if token_ids

        results.append((token_ids, attention_masks, token_type_ids, masks))
        start += args.max_seq_len - 20

    return results


def decode(encodings):
    """

    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :return:
    """
    results = []
    global_index = 0  # 滑窗的指示
    global_indexs = []
    global_values = []
    for (token_ids, attention_masks, token_type_ids, masks) in encodings:
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), 'dev')
        logits = torch.softmax(logits, axis=-1)
        if args.use_crf:
            indexs = model.crf.decode(logits, mask=masks.to(device))[0]
            values = [logits[0, i, j].item() for i, j in enumerate(indexs)]
        else:
            max_logits = torch.max(logits, dim=-1)
            values = max_logits.values.detach().cpu().numpy().tolist()[0]   # 用于置信度判定
            indexs = max_logits.indices.detach().cpu().numpy().tolist()[0]   # 用于label
        # values = [round(float(i), 6) for i in values]
        if len(encodings) == 1:
            global_indexs.extend(copy.copy(indexs))
            global_values.extend(copy.copy(values))
        else:
            if global_index == 0:
                global_indexs.extend(copy.copy(indexs[:-10]))
                global_values.extend(copy.copy(values[:-10]))
                pass
            elif global_index == len(encodings) - 1:
                global_indexs.extend(copy.copy(indexs[10:]))
                global_values.extend(copy.copy(values[10:]))
                pass
            else:
                global_indexs.extend(copy.copy(indexs[10:-10]))
                global_values.extend(copy.copy(values[10:-10]))
            pass
        global_index += 1

    entities = get_entity_bieos([id2label[i] for i in global_indexs])
    for index in range(len(entities)):
        start = entities[index][1] - 1
        end = entities[index][2] + 1
        tags = global_values[start:end]
        confidence = np.mean(np.array(tags))
        entities[index] = entities[index] + (round(float(confidence), 6),)

    return entities


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = './checkpoints/财务附注定位-chinese-roberta-small-wwm-cluecorpussmall-2023-12-05'
args_path = os.path.join(model_name, 'args.json')
model_path = os.path.join(model_name, 'model_best.pt')
labels_path = os.path.join(model_name, 'labels.json')

port = 10089
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(labels_path, 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
# args.gpu_ids = '0'
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
model, device = load_model_and_parallel(LayerLabeling(args), args.gpu_ids, model_path)
model.eval()
for name, param in model.named_parameters():
    param.requires_grad = False
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        # msgs = request.get_json("content")
        msgs = msgs.decode('utf-8')
        msgs = json.loads(msgs)
        assert type(msgs) == list, '输入应为list of str'
        # print(msg)
        # 是否需对句子数量限制 false 是否对单句长度限制 false
        encodings = encode(msgs)
        results = decode(encodings)

        res = json.dumps(results, ensure_ascii=False)
        # torch.cuda.empty_cache()
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port))
    server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
