# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:46
@File    : models.py
@Function: XX
@Other: XX
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from utils.base_model import BaseModel


class LayerFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids, masks, labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.masks = masks
        # labels
        self.labels = labels


class LayerLabeling(BaseModel):
    def __init__(self, args):
        super(LayerLabeling, self).__init__(bert_dir=args.bert_dir,
                                            dropout_prob=args.dropout_prob,
                                            model_name=args.model_name)
        self.args = args
        gpu_ids = self.args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])  # 指示当前设备
        self.args.lstm_hidden = self.base_config.hidden_size // 2

        # 这里num_layers是同一个time_step的结构堆叠 Lstm堆叠层数与time step无关
        self.lstm = nn.LSTM(self.base_config.hidden_size,
                            self.args.lstm_hidden,
                            self.args.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.args.dropout)
        self.linear = nn.Linear(self.args.lstm_hidden * 2, self.args.num_tags)  # lstm之后的线性层
        init_blocks = [self.linear]
        self._init_weights(init_blocks, initializer_range=self.base_config.initializer_range)

        self.crf = CRF(self.args.num_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device,
                         requires_grad=True)
        c0 = torch.randn(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device,
                         requires_grad=True)
        return h0, c0

    def init_hidden_zero(self, batch_size):
        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        h0 = torch.zeros(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device,
                         requires_grad=True)
        c0 = torch.zeros(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device,
                         requires_grad=True)
        return h0, c0

    def forward(self, token_ids, attention_masks, token_type_ids, model='Train'):
        # token_ids: batch_size * max_seq_len * max_word_len
        batch_size = token_ids.size(0)
        max_seq_len = token_ids.size(1)

        output = self.bert_module(input_ids=torch.reshape(token_ids,
                                                          (batch_size * max_seq_len, token_ids.size(-1))),
                                  attention_mask=torch.reshape(attention_masks,
                                                               (batch_size * max_seq_len, attention_masks.size(-1))),
                                  token_type_ids=torch.reshape(token_type_ids,
                                                               (batch_size * max_seq_len, token_type_ids.size(-1))))

        bert_outputs = output[1]
        bert_outputs = torch.reshape(bert_outputs, (batch_size, max_seq_len, self.base_config.hidden_size))

        if model == 'Train':
            hidden = self.init_hidden(batch_size)
        else:
            hidden = self.init_hidden_zero(batch_size)
        seq_out, (hn, _) = self.lstm(bert_outputs, hidden)
        seq_out = seq_out.contiguous().view(-1, self.args.lstm_hidden * 2)
        seq_out = self.linear(seq_out)
        seq_out = seq_out.contiguous().view(batch_size, -1, self.args.num_tags)

        return seq_out
