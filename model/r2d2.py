# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch.nn as nn
from typing import List
import torch
import torch.nn.functional as F
from utils.math_util import gumbel_softmax, max_neg_value
from model.tree_encoder import BinaryEncoder
from model.r2d2_base import R2D2Base
from data_structure.basic_structure import AtomicSpans, DotDict, ChartNode, ChartTable, find_best_merge_batch


class R2D2(R2D2Base):
    def __init__(self, config):
        super().__init__(config)
        self.tree_encoder = BinaryEncoder(config)
        self.blender = nn.Linear(config.hidden_size, 2)
        self.score_linear = nn.Linear(config.hidden_size, 1)
        self.cls_dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                       nn.GELU(),
                                       nn.LayerNorm(config.hidden_size, elementwise_affine=False))

        self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def infer(self, left, right):
        """
        Given representaion of left and right context, and inferring the missing token
        :param left: shape: (batch_size, dim)
        :param right: shape: (batch_size, dim)
        :return: Logits on vocabulary: (batch_size, vocab_size)
        """
        assert left.shape[0] == right.shape[0], "infer(): batch size of left and right doesn't match!"
        bigrams = torch.cat([left.unsqueeze(1), right.unsqueeze(1)], dim=1)
        sz = bigrams.shape[0]
        mask_ids = torch.zeros([sz, ], dtype=torch.long).fill_(self.mask_token_id).to(self.device)
        mask_embedding = self.embedding(mask_ids)  # (sz, hidden_dim)
        input_embedding = torch.cat([mask_embedding.unsqueeze(1), bigrams], dim=1)  # (?, 3, dim)
        outputs = self.tree_encoder(input_embedding.transpose(0, 1)).transpose(0, 1)  # (?, 3, dim)
        mask_hidden = self.norm(outputs[:, 0, :])  # (?, dim)
        return self.classifier(self.cls_dense(mask_hidden))

    def _encode(self, tensor_batch):
        '''
        :param tensor_batch: (?, batch_size, 2, dim)
        :return: representation and log probability for the combinations
                 representation: (?, batch_size, dim)
                 log probability: (?, batch_size)
        '''
        row_len = tensor_batch.shape[0]
        batch_size = tensor_batch.shape[1]
        dim = tensor_batch.shape[-1]
        tensor_batch = tensor_batch.view(-1, 2, dim)
        sz = tensor_batch.shape[0]
        task_ids = torch.tensor([self.cls_token_id, self.sum_token_id], dtype=torch.long) \
            .to(self.device).unsqueeze(0).expand(sz, -1)  # (?, 2)
        tasks_embedding = self.embedding(task_ids)  # (?, 2, dim)
        input_embedding = torch.cat([tasks_embedding, tensor_batch], dim=1)  # (?, 4, dim)
        outputs = self.tree_encoder(input_embedding.transpose(0, 1)).transpose(0, 1)  # (? * batch_size, 4, dim)
        blend_logits = outputs[:, 0, :].view(row_len, batch_size, dim)  # (?, batch_size, dim)
        h_ik = outputs[:, -2, :].view(row_len, batch_size, dim)  # (?, batch_size, dim)
        h_kj = outputs[:, -1, :].view(row_len, batch_size, dim)  # (?, batch_size, dim)
        log_p_ijk = F.logsigmoid(self.score_linear(outputs[:, 1, :]).view(row_len, batch_size))  # (?, batch_size)

        w_ijk = F.softmax(self.blender(blend_logits), dim=-1)  # (?, batch_size, 2)
        h_ik_kj = torch.stack([h_ik, h_kj], dim=3)  # (?, batch_size, dim, 2)
        c_ijk = (h_ik_kj @ w_ijk.unsqueeze(3)).squeeze(-1)  # (?, batch_size, dim)
        return self.norm(c_ijk), log_p_ijk

    @property
    def device(self):
        return next(self.parameters()).device

    def _batch_data(self, chart_tables: List[ChartTable], current_step: int, atomic_spans: List[AtomicSpans]):
        rest_table = 0
        tensor_batches = []
        node_batch = []
        log_p_batch = []

        candidates = []
        total_group = 0
        best_merge_points = find_best_merge_batch(chart_tables, current_step, atomic_spans,
                                                  window_size=self.window_size, device=self.device)
        for table_i, table in enumerate(chart_tables):
            if table.is_finished:
                continue
            rest_table += 1
            if current_step < self.window_size:
                table.expand()
            else:
                i = best_merge_points[table_i]
                table.merge(i)
            table_size = table.table_size
            assert len(table_size) <= self.window_size + 1
            layer = len(table_size) - 1
            assert layer == min(current_step + 1, self.window_size)
            for pos in range(table_size[layer]):
                if table.table(layer, pos).e_ij is None:
                    # check if conflict with atomic spans
                    total_group += 1
                    candidates_local = []
                    for h_i in range(0, layer):
                        assert table.table(h_i, pos).e_ij is not None
                        assert table.table(layer - 1 - h_i, pos + h_i + 1).e_ij is not None
                        tensor_batches.append(table.table(h_i, pos).e_ij)
                        tensor_batches.append(table.table(layer - 1 - h_i, pos + h_i + 1).e_ij)
                        left_cell, right_cell = table.table(h_i, pos), table.table(layer - 1 - h_i, pos + h_i + 1)
                        pair = DotDict()
                        pair.left = left_cell
                        pair.right = right_cell
                        candidates_local.append(pair)
                        log_p_batch.append(table.table(h_i, pos).log_p_sum)
                        log_p_batch.append(table.table(layer - 1 - h_i, pos + h_i + 1).log_p_sum)
                    # batches: (batch_size, 2, dim)
                    candidates.append(candidates_local)
                    node_batch.append(table.table(layer, pos))
        if rest_table > 0:
            tensor_batches = torch.stack(tensor_batches).view(total_group, layer, 2, -1)
            log_p_batch = torch.stack(log_p_batch).view(total_group, layer, 2).sum(dim=2, keepdim=True)

        return rest_table, tensor_batches, node_batch, log_p_batch, candidates

    def forward(self, input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                atom_spans: List[AtomicSpans] = None):
        '''
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :param atom_spans: Default empty. Only used for word level evaluations.
        :return: loss for Bi-LM and encoded chart tables.
        '''
        seq_lens = torch.sum(attention_mask, dim=1)  # (batch_size, 1)
        batch_size = input_ids.shape[0]
        if atom_spans is None:
            atom_spans = [AtomicSpans([]) for _ in range(batch_size)]
        input_embedding = self.embedding(input_ids)  # (batch_size, seq_len, dim)
        chart_tables = []
        for batch_i in range(batch_size):
            nodes = []
            for col in range(seq_lens[batch_i]):
                node = ChartNode(0, col)
                node.e_ij = input_embedding[batch_i][col]
                node._log_p_sum = torch.zeros([1, ]).to(self.device)
                nodes.append(node)
            chart_tables.append(ChartTable(nodes))

        current_step = 0
        while True:
            rest_table, tensor_batches, node_batch, log_p_sum_ik_kj, candidates = \
                self._batch_data(chart_tables, current_step, atomic_spans=atom_spans)
            current_step += 1
            if rest_table == 0:
                break
            c_ijk, log_p_ijk = self._encode(tensor_batches)  # c_ij: (?, batch_size, dim), log_p (?, batch_size)
            log_p_sum_ijk = log_p_sum_ik_kj.squeeze(2) + log_p_ijk  # (?, batch_size)
            a_ij = gumbel_softmax(log_p_sum_ijk, train=self.training)  # (?, batch_size))
            e_ij = c_ijk.permute(0, 2, 1) @ a_ij.unsqueeze(2)  # (?, dim, 1)
            e_ij = e_ij.squeeze(2)  # (?, dim)
            log_p_sum_ij = log_p_sum_ijk.unsqueeze(1) @ a_ij.unsqueeze(2)  # (?, 1, 1)
            log_p_sum_ij = log_p_sum_ij.squeeze(1)
            log_p_ij = log_p_ijk.unsqueeze(1) @ a_ij.unsqueeze(2)
            log_p_ij = log_p_ij.squeeze(1)

            _, s_log_p_sum_indices = log_p_sum_ijk.sort(dim=1, descending=True)
            log_p_sum_ijk = log_p_sum_ijk.unsqueeze(2)
            for node_i, node in enumerate(node_batch):
                node.e_ij = e_ij[node_i]
                for pair_i, pair in enumerate(candidates[node_i]):
                    pair.log_p_sum = log_p_sum_ijk[node_i][pair_i]
                _candidates = [candidates[node_i][_idx] for _idx in s_log_p_sum_indices[node_i]]
                node.candidates = _candidates
                node._log_p_sum = log_p_sum_ij[node_i]
                node.log_p_ij = log_p_ij[node_i]

        bos_vec, eos_vec = (self.bos_vec, self.eos_vec)

        left_batch = []
        right_batch = []
        tgt = []
        for batch_i, seq_len in enumerate(seq_lens):
            for i in range(seq_len):
                left, right = chart_tables[batch_i].gather_tensor(i, bos_vec, eos_vec)
                left_batch.append(left)
                right_batch.append(right)
                tgt.append(input_ids[batch_i, i])
        left_batch = torch.stack(left_batch)
        right_batch = torch.stack(right_batch)
        tgt = torch.stack(tgt)
        logits = self.infer(left_batch, right_batch)
        loss = F.cross_entropy(logits, tgt)
        return loss, chart_tables