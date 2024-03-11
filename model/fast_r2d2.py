# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

import logging
from data_structure.py_backend import CPPChartTableManager
from model.r2d2_base import R2D2Base
from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
from data_structure.tensor_cache import TensorCache, CacheType
from model.r2d2_common import LMLossParam, CacheSlots, SPECIAL_TOKEN_NUM, INF_LOG_P_ID
import model.pretrain_objectives as objectives
from model.tree_encoder import BinaryEncoder
from functools import partial, reduce
from utils.tree_utils import build_trees


logger = logging.getLogger(__name__)


class FastR2D2(R2D2Base):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        # initialize model parameters
        self.span_decoder = False

        self.tree_encoder = BinaryEncoder(config)
        if self.tie_decoder:
            self.tree_decoder = self.tree_encoder
        else:
            self.tree_decoder = BinaryEncoder(config)

        self.blender = nn.Linear(config.hidden_size, 2)

        self._task_ids = None
        self.e_ij_id = -1
        self.score_sum_id = -1

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
                    if 'params' in loss_params:
                        params = loss_params['params']
                    else:
                        params = {}
                    if 'keys' in loss_params:
                        arg_keys = loss_params['keys']
                        for key in arg_keys:
                            params[key] = kwargs[key]
                    self.loss_funcs.append(
                        partial(getattr(objectives, f'cuda_{loss_name}'),
                                **params))
        if len(self.loss_funcs) == 0:
            self.loss_funcs.append(objectives.cuda_default_lm_loss)

    def create_tensor_cache(self, seq_lens, total_cache_size=-1):
        # e_ij, log_p_ij, log_p_sum_ij
        tensor_cache = TensorCache(
            self.window_size,
            seq_lens,
            cache_types=[
                CacheType.NORMAL, CacheType.DETACH
            ],
            dims=[self.input_dim, 1],
            placeholder_num=SPECIAL_TOKEN_NUM,
            device=self.device,
            total_cache_size=total_cache_size)
        self.e_ij_id = 0
        self.score_sum_id = 1
        return tensor_cache

    @property
    def task_ids(self):
        if self._task_ids is None:
            self._task_ids = torch.tensor([self.cls_token_id, self.sum_token_id],
                                            dtype=torch.long,
                                            device=self.device)
        return self._task_ids

    def infer(self, tensor_batch, in_logits=False):
        """
        Given representaion of left and right context, and inferring the missing token
        :param in_logits: indicates whether return logits
        :param tensor_batch: (?, 2, dim)
        :return: Logits on vocabulary: (batch_size, vocab_size)
        """
        sz = tensor_batch.shape[0]
        
        if isinstance(self.tree_decoder, BinaryEncoder):
            mask_ids = torch.zeros([sz,], dtype=torch.long, 
                                    device=self.device).fill_(self.mask_token_id)
            mask_embedding = self.embedding(mask_ids)  # (sz, hidden_dim)
            input_embedding = torch.cat(
                [mask_embedding.unsqueeze(1), tensor_batch], dim=1)  # (?, 3, dim)
            outputs = self.tree_decoder(input_embedding)  # (?, 3, dim)
            mask_hidden = outputs[:, 0, :]  # (?, dim)
        else:
            mask_ids = [[self.mask_token_id] for _ in range(sz)]
            outputs = self.tree_decoder(input_ids = mask_ids, 
                                        memory = tensor_batch, 
                                        embeddings = self.embedding)
            mask_hidden = outputs[:, 0, :]
        if in_logits:
            return self.cls_dense(mask_hidden)
        else:
            return self.classifier(self.cls_dense(mask_hidden))

    def encode(self, tensor_batch, force_encoding=False):
        """
        :param tensor_batch: (?, batch_size, 2, dim)
        :return: representation and log probability for the combinations
                    representation: (?, batch_size, dim)
                    log probability: (?, batch_size)
        """
        row_len = tensor_batch.shape[0]
        batch_size = tensor_batch.shape[1]
        dim = tensor_batch.shape[-1]
        tensor_batch = tensor_batch.view(row_len * batch_size, 2, dim)
        sz = tensor_batch.shape[0]
        # (?, 2)
        if isinstance(self.tree_encoder, BinaryEncoder):
            # (?, 1, dim)
            tasks_embedding = self.embedding(self.task_ids.unsqueeze(0).expand(sz, -1))
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
        else:
            outputs = self.tree_encoder(input_ids_batch = self.task_ids.unsqueeze(0).expand(sz, -1),
                                        memory = tensor_batch,
                                        embeddings = self.embedding)
            c_ijk = outputs[:, 0, :].view(row_len, batch_size, dim)
            log_p_ijk = F.logsigmoid(self.score_linear(outputs[:, 1, :]).view(row_len, batch_size))
            return c_ijk, log_p_ijk
        

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

    def prepare_composition(self, group_ids, log_p_ids, tensor_cache):
        e_ij = tensor_cache.gather(group_ids.flatten(), [self.e_ij_id])[0]
        log_p_ij = tensor_cache.gather(log_p_ids.flatten(), [self.score_sum_id])[0]
        e_ij = e_ij.view(*group_ids.shape, self.input_dim)
        log_p_ij = log_p_ij.view(*group_ids.shape) # (batch_size, group_size, 2)

        return e_ij, log_p_ij.sum(dim=-1)

    def inside(self,
               inside_cache,
               inside_groups):
        splits_orders = []
        
        for target_cache_ids, cache_ids, detach_cache_ids in inside_groups:
            # target_cache_ids: (?)
            # cache_ids: (?, group_size, 2)
            # detach_cache_ids: (?, group_size, 2)

            # if candidate e_ij and log_p is not empty, apply composition function
            e_ij, scores_ij_sum = self.prepare_composition(
                cache_ids, detach_cache_ids, inside_cache)
            # e_ij: (batch_size, group_size, 2, dim), c_ij: (batch_size, 2, dim)

            c_ijk, scores_ijk = self.encode(e_ij)
            # expected output put c_ijk: (batch_size, group_size, dim)
            # log_p_ijk: (batch_size, group_size)

            scores_ijk_sum = scores_ijk + scores_ij_sum  # (batch_size, combination_size)

            # assert not torch.any(torch.isinf(log_p_ij_step))
            a_ij = F.softmax(scores_ijk_sum, dim=-1)
            # (batch_size, combination_size)

            # apply gumbel softmax
            c_ij = torch.einsum("ij,ijk->ik", a_ij, c_ijk)
            
            scores_ij_sum = torch.einsum("ij, ij->i", a_ij, scores_ijk_sum).unsqueeze(1)

            inside_cache.scatter(target_cache_ids, [self.e_ij_id, self.score_sum_id],
                                [c_ij, scores_ij_sum])
            
            splits_orders.append(scores_ijk_sum.argsort(dim=1, descending=True).to('cpu', non_blocking=True))

        return splits_orders

    def forward(self,
                input_ids,
                masks=None,
                merge_trajectories=None,
                atom_spans:List[List[int]]=None,
                recover_tree=False,
                lm_loss=True):
        seq_lens = torch.sum(masks, dim=1,
                             dtype=torch.int)  # (batch_size)
        seq_lens_np = seq_lens.to('cpu', non_blocking=True)
        merge_trajectory = merge_trajectories.to('cpu', non_blocking=True)
        
        flatten_input_ids, input_embedding = \
            self.initialize_embeddings(input_ids, seq_lens)

        ids_num = flatten_input_ids.shape[0]
        input_cache_ids = torch.arange(SPECIAL_TOKEN_NUM, 
                                       SPECIAL_TOKEN_NUM + ids_num).to(self.device)

        inside_cache = self.create_tensor_cache(seq_lens_np)
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
        inside_cache.init_placeholders(
            [CacheSlots.E_IJ, CacheSlots.LOG_P_IJ_SUM],
            [placeholders_embedding, log_p_ij_sum_holders])
        inside_cache.scatter(input_cache_ids, [self.e_ij_id], [input_embedding])
        tables = CPPChartTableManager(seq_lens_np.data.numpy(), self.window_size, merge_trajectory.data.numpy(),
                                      inside_cache.placeholder_num, inside_cache.detach_offset)
        target_cache_ids, cache_ids, detach_cache_ids = \
                tables.construct_inside_groups(self.device)
        root_ids = tables.root_ids

        best_splits = self.inside(inside_cache, 
                                  zip(target_cache_ids, cache_ids, detach_cache_ids))
        root_embedding = inside_cache.gather(root_ids, [self.e_ij_id])[0]

        if flatten_input_ids is not None and lm_loss:
            loss_params = LMLossParam(model=self,
                                      chart_tables=tables,
                                      tensor_cache=inside_cache,
                                      flatten_input_ids=flatten_input_ids,
                                      input_ids=input_ids,
                                      s_indices=merge_trajectory,
                                      atom_spans=atom_spans,
                                      seq_lens=seq_lens_np)
            loss = reduce(lambda a, x: a + x(loss_params), self.loss_funcs, 0)
        else:
            loss = 0

        try:
            result = {"loss": loss}

            result["tensor_cache"] = inside_cache
            result['root_embeddings'] = root_embedding
            if recover_tree:
                targets = torch.full([input_ids.shape[0], 1, input_ids.shape[-1] - 1], fill_value=-1,
                                 requires_grad=False, dtype=torch.long, device=self.device)
                span_masks = torch.full([input_ids.shape[0], 1, input_ids.shape[-1] - 1,
                                    input_ids.shape[-1] - 1], fill_value=0,
                                    requires_grad=False, dtype=torch.int,
                                    device=self.device)  # (batch_size, K, L - 1, L - 1)
                split_ids, cache_ids = tables.best_trees(best_splits, atom_spans)
                trees = build_trees(seq_lens_np, split_ids, cache_ids)
                for batch_i, root in enumerate(trees):
                    visit_queue = [root]
                    tgt = []
                    while len(visit_queue) > 0:
                        current = visit_queue.pop(-1)
                        if current.j - current.i >= 1:
                            span_masks[batch_i, 0, len(tgt), current.i: current.j] = 1
                            tgt.append(current.left.j)
                            visit_queue.append(current.left)
                            visit_queue.append(current.right)
                    targets[batch_i, 0, :len(tgt)] = torch.tensor(tgt)
                
                result['trees'] = [trees, {"split_masks": span_masks, "split_points": targets}]
        except Exception as e:
            raise e
        return result
            
        
