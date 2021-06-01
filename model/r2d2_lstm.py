# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch.nn as nn
from typing import List
import torch
import torch.nn.functional as F

from utils.math_util import gumbel_softmax
from torch.nn import init
from data_structure.basic_structure import AtomicSpans, DotDict, ChartNode, ChartTable, find_best_merge_batch
from collections import namedtuple
from model.r2d2_base import R2D2Base

StateTuple = namedtuple("StateTuple", ("h", "c"))


class R2D2TreeLSTM(R2D2Base):
    """
    For ablation study: replacing kernel composition function by T-LSTM
    """

    def __init__(self, config):
        super().__init__(config)
        input_dim = config.hidden_size
        hidden_dim = config.intermediate_size
        self.W = nn.Parameter(torch.Tensor(5 * hidden_dim, input_dim))
        self.U = nn.Parameter(torch.Tensor(5 * hidden_dim, 2 * hidden_dim))
        self.b = nn.Parameter(torch.Tensor(5 * hidden_dim, ))
        self.output = nn.Sequential(nn.Linear(hidden_dim, input_dim),
                                    nn.GELU())
        self.score_linear = nn.Linear(hidden_dim, 1)

        self._init_LSTM_parameters()

    def _init_LSTM_parameters(self):
        init.uniform_(self.W, -1, 1)
        init.uniform_(self.U, -1, 1)
        init.uniform_(self.b, -1, 1)

    @property
    def bos(self):
        bos_vec = self.embedding(torch.tensor([self.bos_token_id]).to(self.device)).squeeze(0)
        h, c, _ = self.tree_lstm(None, None, None, None, x=bos_vec)
        return StateTuple(h=h, c=c)

    @property
    def eos(self):
        eos_vec = self.embedding(torch.tensor([self.eos_token_id]).to(self.device)).squeeze(0)
        h, c, _ = self.tree_lstm(None, None, None, None, x=eos_vec)
        return StateTuple(h=h, c=c)

    def infer(self, Lh, Rh, Lc, Rc):
        hid = self.hidden_dim
        preact = self.b + torch.einsum('ij, kj->ki', self.U.float(), torch.cat([Lh, Rh], dim=1).float())
        i = torch.sigmoid(preact[:, :hid])
        fL = torch.sigmoid(preact[:, hid:2 * hid] + 1.0)
        fR = torch.sigmoid(preact[:, 2 * hid:3 * hid] + 1.0)
        o = torch.sigmoid(preact[:, 3 * hid:4 * hid])
        u = torch.tanh(preact[:, 4 * hid:])
        c = fL * Lc + fR * Rc + i * u
        h = o * torch.tanh(c)
        return self.classifier(self.output(h))

    def tree_lstm(self, Lh, Rh, Lc, Rc, x=None):
        # Using the same tree_lstm in Mailard's work.
        hid = self.hidden_dim

        if x is None:
            preact = self.b + torch.einsum('ij, klj->kli', self.U.float(), torch.cat([Lh, Rh], dim=2).float())
            i = torch.sigmoid(preact[:, :, :hid])
            fL = torch.sigmoid(preact[:, :, hid:2 * hid] + 1.0)
            fR = torch.sigmoid(preact[:, :, 2 * hid:3 * hid] + 1.0)
            o = torch.sigmoid(preact[:, :, 3 * hid:4 * hid])
            u = torch.tanh(preact[:, :, 4 * hid:])
            c = fL * Lc + fR * Rc + i * u
        else:
            preact = self.b + self.W @ x
            i = torch.sigmoid(preact[:hid])
            o = torch.sigmoid(preact[3 * hid:4 * hid])
            u = torch.tanh(preact[4 * hid:])
            c = i * u

        h = o * torch.tanh(c)

        log_p = F.logsigmoid(self.score_linear(h))

        return h, c, log_p

    @property
    def device(self):
        return next(self.parameters()).device

    def _batch_data(self, chart_tables, current_step, atomic_spans):
        rest_table = 0
        Lh = []
        Rh = []
        Lc = []
        Rc = []
        node_batch = []
        log_p_batch = []
        candidates = []
        best_merge_points = find_best_merge_batch(chart_tables, current_step, atomic_spans, self.window_size, self.device)
        for table_i, table in enumerate(chart_tables):
            if table.is_finished:
                continue
            rest_table += 1
            if current_step < self.window_size:
                table.expand()
            else:
                i = best_merge_points[table_i]
                table.merge(i)
            # list tensors to encode
            table_size = table.table_size
            assert len(table_size) <= self.window_size + 1
            layer = len(table_size) - 1
            assert layer == min(current_step + 1, self.window_size)
            for pos in range(table_size[layer]):
                if table.table(layer, pos).h is None:
                    Lh_batch = []
                    Rh_batch = []
                    Lc_batch = []
                    Rc_batch = []
                    left_p_batch = []
                    right_p_batch = []
                    candidates_local = []
                    for h_i in range(0, layer):
                        assert table.table(h_i, pos).h is not None and table.table(h_i, pos).c is not None
                        assert table.table(layer - 1 - h_i, pos + h_i + 1).h is not None
                        assert table.table(layer - 1 - h_i, pos + h_i + 1).c is not None
                        Lh_batch.append(table.table(h_i, pos).h)
                        Rh_batch.append(table.table(layer - 1 - h_i, pos + h_i + 1).h)
                        Lc_batch.append(table.table(h_i, pos).c)
                        Rc_batch.append(table.table(layer - 1 - h_i, pos + h_i + 1).c)
                        pair = DotDict()
                        pair.left = table.table(h_i, pos)
                        pair.right = table.table(layer - 1 - h_i, pos + h_i + 1)
                        candidates_local.append(pair)
                        left_p_batch.append(table.table(h_i, pos).log_p_sum)
                        right_p_batch.append(table.table(layer - 1 - h_i, pos + h_i + 1).log_p_sum)
                    assert len(Lh_batch) == layer
                    candidates.append(candidates_local)
                    Lh.append(torch.stack(Lh_batch))
                    Rh.append(torch.stack(Rh_batch))
                    Lc.append(torch.stack(Lc_batch))
                    Rc.append(torch.stack(Rc_batch))
                    log_p_batch.append(torch.stack(left_p_batch) + torch.stack(right_p_batch))
                    node_batch.append(table.table(layer, pos))

        return rest_table, Lh, Rh, Lc, Rc, node_batch, log_p_batch, candidates

    def forward(self, input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                atom_spans: List[AtomicSpans] = None,
                output_weights=False):
        '''
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :param atom_spans:
        :param output_weights:
        :return:
        '''
        seq_lens = torch.sum(attention_mask, dim=1)  # (batch_size, 1)
        max_len = max(seq_lens)
        batch_size = input_ids.shape[0]
        if atom_spans is None:
            atom_spans = [AtomicSpans([]) for _ in range(batch_size)]
        input_embedding = self.embedding(input_ids)  # (batch_size, seq_len, dim)
        input_embedding_chunked = input_embedding.transpose(0, 1).chunk(max_len, dim=0)  # (seq_len, batch_size, dim)
        chart_tables = []
        input_list = [v.squeeze(0) for v in input_embedding_chunked]  # List([batch_size, dim]

        def create_node_func(height, pos, vec):
            return ChartNode(height, pos, vec, extra_attributes=['h', 'c'])

        for batch_i in range(batch_size):
            nodes = []
            for col in range(max_len):
                node = ChartNode(0, col, extra_attributes=['h', 'c'])
                h, c, _ = self.tree_lstm(None, None, None, None, input_list[col][batch_i])
                node.h = h
                node.c = c
                node._log_p_sum = torch.zeros([1, ]).to(self.device)
                nodes.append(node)
            chart_tables.append(ChartTable(nodes, create_node_func=create_node_func))

        current_step = 0
        while True:
            rest_table, Lh, Rh, Lc, Rc, node_batch, log_p_batch, candidates = \
                self._batch_data(chart_tables, current_step, atom_spans)
            current_step += 1
            if rest_table == 0:
                break
            log_p_batch = torch.stack(log_p_batch)  # (?, batch_size, 1)
            h, c, log_p_ijk = self.tree_lstm(torch.stack(Lh), torch.stack(Rh),
                                                     torch.stack(Lc), torch.stack(Rc))  # h (?, batch_size, dim)
            log_p_batch = log_p_batch.squeeze(2) + log_p_ijk.squeeze(2)
            # gumbel softmax
            weight = gumbel_softmax(log_p_batch, train=self.training)  # (?, batch_size))
            h = h.permute(0, 2, 1) @ weight.unsqueeze(2)  # (?, dim, 1)
            h = h.squeeze(2)  # (?, dim)
            c = c.permute(0, 2, 1) @ weight.unsqueeze(2)  # (?, dim, 1)
            c = c.squeeze(2)  # (?, dim)
            node_p_batch = log_p_batch.unsqueeze(1) @ weight.unsqueeze(2)  # (?, 1, 1)
            node_p_batch = node_p_batch.squeeze(1)
            log_p_ij = log_p_ijk.permute(0, 2, 1) @ weight.unsqueeze(2)
            log_p_ij = log_p_ij.squeeze(1)

            log_p_batch = log_p_batch.unsqueeze(2)
            for node_i, node in enumerate(node_batch):
                node.h = h[node_i]
                node.c = c[node_i]
                for pair_i, pair in enumerate(candidates[node_i]):
                    pair.log_p_sum = log_p_batch[node_i][pair_i]
                candidates[node_i].sort(key=lambda x: x.log_p_sum, reverse=True)
                node.candidates = candidates[node_i]
                node._log_p_sum = node_p_batch[node_i]
                node.log_p_ij = log_p_ij[node_i]

        Lh_batch = []
        Rh_batch = []
        Lc_batch = []
        Rc_batch = []
        tgt = []
        for batch_i, seq_len in enumerate(seq_lens):
            for i in range(seq_len):
                left, right = chart_tables[batch_i].gather_node(i, self.bos, self.eos)
                Lh_batch.append(left.h)
                Rh_batch.append(right.h)
                Lc_batch.append(left.c)
                Rc_batch.append(right.c)
                tgt.append(input_ids[batch_i, i])
        Lh_batch = torch.stack(Lh_batch)
        Rh_batch = torch.stack(Rh_batch)
        Lc_batch = torch.stack(Lc_batch)
        Rc_batch = torch.stack(Rc_batch)
        tgt = torch.stack(tgt)
        logits = self.infer(Lh_batch, Rh_batch, Lc_batch, Rc_batch)
        loss = F.cross_entropy(logits, tgt)
        return loss, chart_tables