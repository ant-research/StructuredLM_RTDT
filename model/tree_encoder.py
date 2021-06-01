# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import deepcopy


ACTIVATION_POOL = ['relu', 'gelu']


def _get_activation_fn(activation):
    if activation in ACTIVATION_POOL:
        return getattr(F, activation)

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TreeEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_role_count, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_role_count, d_model)

        self.activation = _get_activation_fn(activation)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, src, src_mask=None, pos_ids=None):
        """
        :param src: concatenation of task embeddings and representation for left and right.
                    src shape: (task_embeddings + left + right, batch_size, dim)
        :param src_mask:
        :param pos_ids:
        :return:
        """
        if src_mask is None:
            task_count = src.shape[0] - 2
            pos_ids = torch.tensor([0] * task_count + [1, 2], dtype=torch.long).to(self.device)
            src_mask = [[float('-inf') for _ in range(task_count + 2)] for _ in range(task_count + 2)]
            for pos_i in range(task_count + 2):
                if pos_i < task_count:
                    src_mask[pos_i][pos_i] = 0
                src_mask[pos_i][-1] = 0
                src_mask[pos_i][-2] = 0

            src_mask = torch.tensor(src_mask).to(self.device)
        if len(pos_ids.shape) == 1:
            sz = src.shape[1]  # sz: batch_size
            pos_ids = pos_ids.unsqueeze(1).expand(-1, sz)  # (3, batch_size)
        position_embedding = self.position_embedding(pos_ids)
        src2 = self.self_attn(src + position_embedding, src + position_embedding, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BinaryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = TreeEncoderLayer(config.hidden_size,
                                 config.num_attention_heads,
                                 config.intermediate_size,
                                 max_role_count=config.max_role_embeddings,
                                 dropout=config.attention_probs_dropout_prob,
                                 activation='gelu')
        self.layers = nn.ModuleList([layer] + [deepcopy(layer) for _ in range(config.encoder_num_hidden_layers - 1)])

    def forward(self, src, src_mask=None, pos_ids=None):
        """
        :param pos_ids:
        :param src_mask:
        :param src:
        :return:
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask, pos_ids)

        return output