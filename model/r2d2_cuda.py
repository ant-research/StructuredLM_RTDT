# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import logging

from model.r2d2_base import R2D2Base
from typing import List, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from data_structure.tensor_cache import TensorCache, CacheType
from model.r2d2_common import LMLossParam, CacheSlots, SPECIAL_TOKEN_NUM, INF_LOG_P_ID
import model.pretrain_objectives as objectives
from model.tree_encoder import BinaryEncoder
from functools import partial, reduce
from utils.math_util import gumbel_softmax
from utils.table_converter import convert_cuda_tables
import r2d2lib


logger = logging.getLogger(__name__)


class R2D2Cuda(R2D2Base):
    def __init__(self, config):
        super().__init__(config)

        # initialize model parameters
        self.tree_encoder = BinaryEncoder(config)
        if self.tie_decoder:
            self.tree_decoder = self.tree_encoder
        else:
            self.tree_decoder = BinaryEncoder(config)

        self.blender = nn.Linear(config.hidden_size, 2)

        self._task_ids = None

        self.score_linear = nn.Linear(config.hidden_size, 1)
        self.cls_dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.attention_probs_dropout_prob)
        )

        if hasattr(config, "max_feature_num"):
            if config.max_feature_num is not None:
                self.feature_embeddings = nn.Embedding(config.max_feature_num,
                                                    config.embedding_dim)

        self.norm = nn.InstanceNorm1d(config.hidden_size)

        # initialize loss functions
        self.loss_funcs = []
        if hasattr(config, "loss"):
            if config.loss is not None:
                for loss_params in config.loss:
                    loss_name = loss_params['name']
                    loss_params = loss_params['params']
                    self.loss_funcs.append(
                        partial(getattr(objectives, f'cuda_{loss_name}'),
                                **loss_params))
            else:
                self.loss_funcs.append(objectives.cuda_default_lm_loss)

    def create_tensor_cache(self, seq_lens):
        # e_ij, log_p_ij, log_p_sum_ij
        tensor_cache = TensorCache(
            self.window_size,
            seq_lens,
            cache_types=[
                CacheType.NORMAL, CacheType.DETACH
            ],
            dims=[self.input_dim, 1],
            placeholder_num=SPECIAL_TOKEN_NUM,
            device=self.device)
        return tensor_cache

    def infer(self, tensor_batch, in_logits=False):
        """
        Given representaion of left and right context, and inferring the missing token
        :param in_logits: indicates whether return logits
        :param tensor_batch: (?, 2, dim)
        :return: Logits on vocabulary: (batch_size, vocab_size)
        """
        sz = tensor_batch.shape[0]
        mask_ids = torch.zeros([
            sz,
        ], dtype=torch.long, device=self.device).fill_(self.mask_token_id)
        mask_embedding = self.embedding(mask_ids)  # (sz, hidden_dim)
        input_embedding = torch.cat(
            [mask_embedding.unsqueeze(1), tensor_batch], dim=1)  # (?, 3, dim)
        outputs = self.tree_decoder(input_embedding)  # (?, 3, dim)
        mask_hidden = outputs[:, 0, :]  # (?, dim)
        if in_logits:
            return self.cls_dense(mask_hidden)
        else:
            return self.classifier(self.cls_dense(mask_hidden))

    @property
    def task_ids(self):
        if self._task_ids is None:
            self._task_ids = torch.tensor([self.cls_token_id, self.sum_token_id],
                                          dtype=torch.long,
                                          device=self.device)
        return self._task_ids

    def encode(self, tensor_batch):
        """
        :param tensor_batch: (?, batch_size, 2, dim)
        :return: representation and log probability for the combinations
                    representation: (?, batch_size, dim)
                    log probability: (?, batch_size)
        """
        # row_len = tensor_batch.shape[0]
        # batch_size = tensor_batch.shape[1]
        # dim = tensor_batch.shape[-1]
        # result = self.mlp_encoder(
        #     tensor_batch.view(row_len, batch_size, 2 * dim))
        # score = F.logsigmoid(self.mlp_scorer(result))
        # return result, score.squeeze(2)

        row_len = tensor_batch.shape[0]
        batch_size = tensor_batch.shape[1]
        dim = tensor_batch.shape[-1]
        tensor_batch = tensor_batch.view(row_len * batch_size, 2, dim)
        sz = tensor_batch.shape[0]

        # (?, 1)
        tasks_embedding = self.embedding(self.task_ids.unsqueeze(0).expand(sz, -1))
        # (?, 1, dim)
        input_embedding = torch.cat([tasks_embedding, tensor_batch],
                                    dim=1)  # (?, 4, dim)
        outputs = self.tree_encoder(
            input_embedding)  # (? * batch_size, 4, dim)

        log_p_ijk = F.logsigmoid(
            self.score_linear(outputs[:, 0, :]).view(
                row_len, batch_size))  # (?, batch_size)

        logits = outputs[:, 1, :].view(row_len, batch_size, dim)  # (?, batch_size, dim)
        w_ijk = F.softmax(self.blender(logits), dim=-1)  # (?, batch_size, 2)
        h_ik_kj = outputs[:, -2:, :].view(row_len, batch_size, 2, dim)
        c_ijk = torch.einsum("ijk,ijk...->ij...", w_ijk, h_ik_kj)  # (?, batch_size, dim)

        return self.norm(c_ijk), log_p_ijk

    def initialize_embeddings(self, input_ids, seq_lens, input_embeddings=None,
                               feature_ids_list=None, tensor_cache = None,
                               id_offset = 0):
        # Initialize embeddings
        block_size = input_ids.shape[
            -1] if input_ids is not None else input_embeddings.shape[1]
        indices_gather = []
        for seq_i, seq_len in enumerate(seq_lens):
            indices_gather.extend(
                range(block_size * seq_i, block_size * seq_i + seq_len))
        if input_ids is not None:
            flatten_input_ids = input_ids.flatten()
            flatten_input_ids = flatten_input_ids.gather(
                dim=0, index=torch.tensor(indices_gather, device=self.device))
            embeddings = self.embedding(flatten_input_ids)
        else:
            flatten_input_ids = None
            input_embeddings_flatten = input_embeddings.view(
                -1, input_embeddings.shape[-1])
            embeddings = input_embeddings_flatten.index_select(
                dim=0, index=torch.tensor(indices_gather, device=self.device))
        if feature_ids_list is not None:
            for feature_ids in feature_ids_list:
                flatten_feature_ids = feature_ids.flatten()
                flatten_feature_ids = flatten_feature_ids.gather(
                    dim=0,
                    index=torch.tensor(indices_gather, device=self.device))
                embeddings += self.feature_embeddings(flatten_feature_ids)
        padding_zeros = torch.zeros(len(indices_gather), 1, device=self.device)
        if tensor_cache is not None:
            tensor_cache.fill(id_offset,
                            sum(seq_lens),
                            cache_ids=[CacheSlots.E_IJ, CacheSlots.LOG_P_IJ_SUM],
                            values=[embeddings, padding_zeros])
        return flatten_input_ids, embeddings

    def _topk(self, group_ids, log_p_ids, tensor_cache, combination_size):
        e_ij = tensor_cache.gather(group_ids.flatten(),
                                   [CacheSlots.E_IJ])[0]
        log_p_ij = tensor_cache.gather(log_p_ids.flatten(),
                                       [CacheSlots.LOG_P_IJ_SUM])[0]
        e_ij = e_ij.view(*group_ids.shape, self.input_dim)
        log_p_ij = log_p_ij.view(*group_ids.shape)

        indices = torch.arange(0, combination_size).unsqueeze(0).repeat(
            group_ids.shape[0], 1)
        return indices, e_ij, log_p_ij.sum(dim=-1), None

    def _hierarchical_encoding(self, tables, cache_ids, log_p_ids,
                               bigram_scores, span_lens, tensor_cache):
        """
        span_lens: [total_cell]
        """
        candidates_log_p_pool = torch.full([tensor_cache.capacity,
                                            self.window_size], -1e7, device=self.device)
        log_p_offset = tables.total_len()
        while not tables.finished():
            bigram_scores.fill_(float("-inf"))  # (batch_size, max_len)
            if self.training:
                noise = -torch.empty_like(
                    bigram_scores,
                    memory_format=torch.legacy_contiguous_format,
                    requires_grad=False).exponential_().log()
            else:
                noise = torch.zeros_like(bigram_scores, requires_grad=False)
            span_lens = torch.ones_like(span_lens,
                                        dtype=torch.float,
                                        requires_grad=False)

            if tables.current_step() > self.window_size:
                apply_size = 2 * (self.window_size + 1) * bigram_scores.shape[0] * self.window_size
                cache_ids = torch.full([apply_size],
                                       INF_LOG_P_ID, requires_grad=False,
                                       dtype=torch.int, device=self.device)
                log_p_ids = torch.full([apply_size], tensor_cache.detach_offset, requires_grad=False,
                                       dtype=torch.int, device=self.device)
            else:
                cache_ids = torch.full(cache_ids.shape, INF_LOG_P_ID, requires_grad=False,
                                       dtype=torch.int, device=self.device)
                log_p_ids = torch.full(log_p_ids.shape, tensor_cache.detach_offset, requires_grad=False,
                                       dtype=torch.int, device=self.device)

            batch_size, current_size, group_size, cache_id_offset, ids_len = tables.step(
                cache_ids, log_p_ids, span_lens, bigram_scores, noise)

            indices_selected, e_ij, log_p_ij_sum, candidates_log_p = self._topk(
                cache_ids[:batch_size * group_size * 2].view(batch_size, group_size, 2),
                log_p_ids[:batch_size * group_size * 2].view(batch_size, group_size, 2),
                tensor_cache, current_size)

            c_ijk, log_p_ijk = self.encode(e_ij)

            log_p_ijk_sum = log_p_ijk + log_p_ij_sum  # (batch_size, combination_size)

            # assert not torch.any(torch.isinf(log_p_ij_step))
            a_ij = gumbel_softmax(log_p_ijk_sum)
            # (batch_size, combination_size)

            # apply gumbel softmax
            candidates_log_p = log_p_ijk_sum  # (batch_size, depth)
            c_ij = torch.einsum("ij,ijk->ik", a_ij, c_ijk)
            log_p_ij_sum = torch.einsum("ij, ij->i", a_ij,
                                        log_p_ijk_sum).unsqueeze(1)
            _, indices_selected = a_ij.max(dim=-1)
            indices_selected = indices_selected.unsqueeze(1)

            tables.beam_select(indices_selected)
            tensor_cache.fill(
                cache_id_offset,
                ids_len,
                cache_ids=[CacheSlots.E_IJ, CacheSlots.LOG_P_IJ_SUM],
                values=[c_ij, log_p_ij_sum])

            tables.step_over(log_p_ij_sum, candidates_log_p)
            candidates_log_p_pool[log_p_offset: log_p_offset + batch_size, :candidates_log_p.shape[1]] = \
                candidates_log_p
            log_p_offset += batch_size
        return candidates_log_p_pool

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None,
                feature_ids_list: List[torch.Tensor] = None,
                merge_trajectories: torch.Tensor = None,
                input_embeddings: torch.Tensor = None,
                recover_tree: bool = False,
                keep_tensor_cache: bool = False,
                sample_trees: int = -1,
                lm_loss: bool = True):
        """
        :param sample_trees: If train a parser, specify the num of trees to sample
        :param merge_trajectories: The merge order during pruning.
        :param keep_tensor_cache: Flag indicates whether the tensor cache used in encoding is returned
        :param feature_ids_list: list of (batch_size, seq_len): add extra feature ids
        :param recover_tree: Indicator for whether encoding actions are kept
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :return: Dictionary contains encoding results, bilm loss(lm_loss=True), trees(recover_tree=True), 
                sampled trees(sample_trees>0).
        """
        seq_lens = torch.sum(attention_mask, dim=1,
                             dtype=torch.int)  # (batch_size, 1)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeddings.shape[0]

        if feature_ids_list is not None:
            assert hasattr(self, "feature_embeddings")
            for feature_ids in feature_ids_list:
                if input_ids is not None:
                    assert feature_ids.shape == input_ids.shape

        placeholders_embedding = torch.full(
            [SPECIAL_TOKEN_NUM, self.input_dim],
            0.0,
            dtype=torch.float,
            device=self.device)
        placeholders_embedding[0:2] = self.embedding(
            torch.tensor(
                [self.bos_token_id, self.eos_token_id],
                device=self.device))
        log_p_ij_sum_holders = torch.full([SPECIAL_TOKEN_NUM, 1],
                                          0.0,
                                          dtype=torch.float,
                                          device=self.device)
        log_p_ij_sum_holders[INF_LOG_P_ID] = -1e7  # float('-inf')
        seq_lens_np = seq_lens.to("cpu").numpy()
        tensor_cache = self.create_tensor_cache(seq_lens_np)
        tensor_cache.init_placeholders(
            [CacheSlots.E_IJ, CacheSlots.LOG_P_IJ_SUM],
            [placeholders_embedding, log_p_ij_sum_holders])

        tables = r2d2lib.TablesManager(False, self.window_size, 1)
        tables.encoding_start(seq_lens, SPECIAL_TOKEN_NUM,
                              tensor_cache.detach_offset, INF_LOG_P_ID)
        if merge_trajectories is not None:
            tables.set_merge_trajectories(merge_trajectories)

        # cache_id tensor
        cache_ids = torch.full([sum(seq_lens) * self.window_size * 2],
                               0,
                               dtype=torch.int,
                               requires_grad=False,
                               device=self.device)
        log_p_ids = torch.full([sum(seq_lens) * self.window_size * 2],
                               0,
                               dtype=torch.int,
                               requires_grad=False,
                               device=self.device)
        bigram_scores = torch.full(
            [len(seq_lens), max(seq_lens)],
            0.0,
            dtype=torch.float,
            requires_grad=False,
            device=self.device)
        span_lens = torch.full([sum(seq_lens_np)],
                               1.0,
                               dtype=torch.float,
                               requires_grad=False,
                               device=self.device)
        batch_size, _, _, _, _ = tables.step(cache_ids, log_p_ids, span_lens,
                                             bigram_scores,
                                             torch.zeros_like(bigram_scores))
        # batch_size = sum(seq_lens)
        # fill token embedding to tensor cache
        flatten_input_ids, _ = self.initialize_embeddings(
            input_ids, seq_lens_np, input_embeddings, feature_ids_list,
            tensor_cache, SPECIAL_TOKEN_NUM)
        log_p_batch = torch.full([batch_size, 1],
                                 0.0,
                                 dtype=torch.float,
                                 device=self.device)
        tables.step_over(log_p_batch, log_p_batch)

        candidates_log_p = self._hierarchical_encoding(tables, cache_ids, log_p_ids,
                                                       bigram_scores, span_lens, tensor_cache)

        if flatten_input_ids is not None and lm_loss:
            loss_params = LMLossParam(model=self,
                                      chart_tables=tables,
                                      tensor_cache=tensor_cache,
                                      flatten_input_ids=flatten_input_ids)
            loss = reduce(lambda a, x: a + x(loss_params), self.loss_funcs, 0)
        else:
            loss = 0

        try:
            result = {"loss": loss}
            if recover_tree:
                result["tables"] = convert_cuda_tables(tables.dump_cells(), tensor_cache)
            if keep_tensor_cache:
                result["tensor_cache"] = tensor_cache
            if sample_trees > 0:
                split_points = F.softmax(candidates_log_p, dim=-1).multinomial(num_samples=sample_trees,
                                                                               replacement=True)

                assert torch.all(~torch.isnan(F.softmax(candidates_log_p, dim=-1)))
                # (total_len, num_samples)
                span_masks = torch.full([input_ids.shape[0], sample_trees, input_ids.shape[-1] - 1,
                                         input_ids.shape[-1] - 1], fill_value=0,
                                        requires_grad=False, dtype=split_points.dtype,
                                        device=self.device)  # (batch_size, K, L - 1, L - 1)
                targets = torch.full([input_ids.shape[0], sample_trees, input_ids.shape[-1] - 1], fill_value=-1,
                                     requires_grad=False, dtype=split_points.dtype, device=self.device)
                tables.recover_sampled_trees(span_masks, targets, split_points)
                result["sampled_trees"] = {"split_masks": span_masks, "split_points": targets}
            return result
        except Exception as e:
            raise e
        finally:
            tables.encoding_over()
