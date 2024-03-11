# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

from collections import namedtuple
import logging
from typing import List
from data_structure.py_backend import CPPChartTableManager
from model.r2d2_base import R2D2Base
import torch.nn as nn
import torch
import torch.nn.functional as F
from data_structure.tensor_cache import TensorCache, CacheType
from model.r2d2_common import SPECIAL_TOKEN_NUM
from model.inside_outside_residual import InsideOutsideLayerEncoder
from functools import partial
import numpy as np

from utils.tree_utils import build_trees, rebuild_tgt_ids
from .fast_r2d2_functions import pairwise_contextual_inside_outside, rebuild_batch_indices_cpu

logger = logging.getLogger(__name__)


InsideGroup = namedtuple("InsideGroup", ["parent_ids", "candidate_e_ij_ids", "candidate_log_p_ids", 
                                         "idx2batch", "span_lens"])


class FastR2D2Plus(R2D2Base):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.iter_times = config.iter_times
        
        self.inout_encoders = nn.ModuleList([InsideOutsideLayerEncoder(config) for _ in range(self.iter_times)])
        self.inside_layers = [self.inout_encoders[iter].inside for iter in range(self.iter_times)]
        self.outside_layers = [self.inout_encoders[iter].outside for iter in range(self.iter_times)]
        enc_layer = nn.TransformerEncoderLayer(config.hidden_size, 
                                               config.num_attention_heads,
                                               config.intermediate_size,
                                               activation=F.gelu,
                                               batch_first=True,
                                               norm_first=True)
        self.span_self_attention = nn.TransformerEncoder(enc_layer, config.span_attention_num_layers)
        
        self.empty_cell = nn.Parameter(torch.rand(config.embedding_dim))
        self.cls_embedding = nn.Parameter(torch.rand(config.embedding_dim))

        self.head_num = config.span_num_heads

        self.cls_dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.attention_probs_dropout_prob)
        )

        self.e_ij_id = -1
        self.score_sum_id = -1
        self.score_mean_id = -1
        self.score_ijk = -1

        if hasattr(config, "max_feature_num"):
            if config.max_feature_num is not None:
                self.feature_embeddings = nn.Embedding(config.max_feature_num,
                                                       config.embedding_dim)

        self.norm = nn.InstanceNorm1d(config.hidden_size)
            
    def create_tensor_cache(self, seq_lens, total_cache_size=-1):
        # e_ij, log_p_ij, log_p_sum_ij
        tensor_cache = TensorCache(
            self.window_size,
            seq_lens,
            cache_types=[
                CacheType.NORMAL, CacheType.DETACH,
                CacheType.NORMAL, CacheType.NORMAL
            ],
            dims=[self.input_dim, 1, 1, self.window_size],
            placeholder_num=SPECIAL_TOKEN_NUM,
            device=self.device,
            total_cache_size=total_cache_size)
        self.e_ij_id = 0
        self.score_sum_id = 1
        self.score_mean_id = 2
        self.score_ijk = 3
        return tensor_cache
    
    def initialize_embeddings(self, input_ids, seq_lens, tgt_ids=None):
            # Initialize embeddings
        block_size = input_ids.shape[-1]
        indices_gather = []
        for seq_i, seq_len in enumerate(seq_lens):
            indices_gather.extend(
                range(block_size * seq_i, block_size * seq_i + seq_len))
            
        if input_ids is not None:
            flatten_input_ids = input_ids.flatten()
            flatten_input_ids = flatten_input_ids.gather(
                dim=0, index=torch.tensor(indices_gather, device=self.device))
            flatten_tgt_ids = None
            if tgt_ids is not None:
                flatten_tgt_ids = tgt_ids.flatten()
                flatten_tgt_ids = flatten_tgt_ids.gather(
                    dim=0, index=torch.tensor(indices_gather, device=self.device))
            embeddings = self.embedding(flatten_input_ids)
            
        return flatten_input_ids, flatten_tgt_ids, embeddings

    def prepare_composition(self, group_ids, log_p_ids, tensor_cache):
        e_ij = tensor_cache.gather(group_ids.flatten(), [self.e_ij_id])[0]
        log_p_ij = tensor_cache.gather(log_p_ids.flatten(), [self.score_sum_id])[0]
        e_ij = e_ij.view(*group_ids.shape, self.input_dim)
        log_p_ij = log_p_ij.view(*group_ids.shape) # (batch_size, group_size, 2)

        return e_ij, log_p_ij.sum(dim=-1)

    def inside(self,
               inout_encoder,
               inside_cache,
               inside_groups,
               outside_cache=None,
               is_final_layer=False):
        splits_orders = []
        
        for target_cache_ids, cache_ids, detach_cache_ids in inside_groups:
            # target_cache_ids: (?)
            # cache_ids: (?, group_size, 2)
            # detach_cache_ids: (?, group_size, 2)
            if outside_cache is not None:
                parent_ij = outside_cache.gather(target_cache_ids, [self.e_ij_id])[0]
            else:
                parent_ij = self.empty_cell.unsqueeze(0).repeat(target_cache_ids.shape[0], 1)
            # if candidate e_ij and log_p is not empty, apply composition function
            e_ij, scores_ij_sum = self.prepare_composition(
                cache_ids, detach_cache_ids, inside_cache)
            # e_ij: (batch_size, group_size, 2, dim), c_ij: (batch_size, 2, dim)

            scores_ijk, c_ijk = inout_encoder.inside(e_ij, parent_ij)
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
            
            if is_final_layer:
                # padding to group_size
                splits_orders.append(scores_ijk_sum.argsort(dim=1, descending=True).to('cpu', non_blocking=True))

        return splits_orders
        
    
    def outside(self, inout_encoder, batch_size, root_ids, root_embedding, inside_cache, outside_groups):
        # initialize tensor cache for outside algorithm
        out_cache_size = inside_cache.capacity - inside_cache.placeholder_num
        outside_cache = TensorCache(0, None, [CacheType.NORMAL, CacheType.NORMAL, CacheType.NORMAL],
                                    [self.input_dim, 1, 1], inside_cache.placeholder_num,
                                    total_cache_size=out_cache_size, 
                                    device=inside_cache.device)
        topdown_e_ij_slot = 0
        topdown_score_slot = 1  # weighted sum for outside scores
        topdown_score_ln_sum = 2  # store log (e^w1 + e^w2 + e^w3), w1, w2, w3 is the calculated outside scores
        
        # (batch_size, dim), add root role embedding
        
        zero_padding = torch.zeros(batch_size, 1, dtype=torch.float, device=self.device)
        neg_padding = torch.zeros((outside_cache.capacity, 1), dtype=torch.float, device=self.device).fill_(-1e20)
        
        # As there is no calcuated outside scores, initialize caches with a huge neg value
        outside_cache.fill(0, outside_cache.capacity, [topdown_score_ln_sum], [neg_padding])
        outside_cache.scatter(root_ids.long(), [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                              [root_embedding, zero_padding, zero_padding])

        # run outside according to inside groups
        for target_cache_ids, cache_ids, _ in outside_groups:
            parent_ids = target_cache_ids
            child_ids = cache_ids
            
            parent_ij, parent_ij_score = outside_cache.gather(parent_ids, [topdown_e_ij_slot, topdown_score_slot])

            child_ids_shape = child_ids.shape  # (batch_size, comb_size, 2)
            child_ikj, child_scores = inside_cache.gather(child_ids.flatten(), [self.e_ij_id, self.score_sum_id])
            child_ikj = child_ikj.view(*child_ids.shape, -1)
            child_scores = child_scores.view(*child_ids.shape)  # (batch_size, comb_size, 2)

            out_scores, out_ikj = inout_encoder.outside(parent_ij, child_ikj, parent_ij_score, child_scores)
            # span_norm = (1 + max_lens - span_lens).unsqueeze(1).unsqueeze(2)
            # out_ikj: (batch_size, comb_size, 2)
            
            dim = out_ikj.shape[-1]

            # print(f"out scores: {out_scores[:5, :, :]}, out ikj: {out_ikj[:5, :, :, :5]}")
            # weighted sum left and right seperately
            weighted_e_ij, weighted_scores, log_ksum_score = \
                outside_cache.gather(child_ids[:, :, 0].flatten(), 
                                     [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum])
            weighted_e_ij = weighted_e_ij.view(*child_ids_shape[:-1], dim)  # (batch_size, comb_size, dim)
            log_ksum_score = log_ksum_score.view(*child_ids_shape[:-1])  # (batch_size, comb_size)
            weighted_scores = weighted_scores.view(*child_ids_shape[:-1])

            # log_p_ijk_mean: (batch_size, comb_size)
            left_k_sum_scores = torch.stack([log_ksum_score, out_scores[:, :, 0]], dim=2)  # (batch_size, comb_size, 2)
            left_k_weights = F.softmax(left_k_sum_scores, dim=2)
            left_weighted_e_ij = left_k_weights[:, :, 0].unsqueeze(2) * weighted_e_ij + \
                                    left_k_weights[:, :, 1].unsqueeze(2) * out_ikj[:, :, 0, :]
            left_weighted_scores = left_k_weights[:, :, 0] * weighted_scores + \
                                    left_k_weights[:, :, 1] * out_scores[:, :, 0]

            # (batch_size, comb_size, dim)
            left_k_sum_scores = left_k_sum_scores.logsumexp(dim=2, keepdim=True)

            left_weighted_e_ij = left_weighted_e_ij.view(-1, dim)
            left_weighted_scores = left_weighted_scores.view(-1, 1)
            left_k_sum_scores = left_k_sum_scores.view(-1, 1)

            outside_cache.scatter(child_ids[:, :, 0].flatten().long(), 
                                  [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                                  [left_weighted_e_ij, left_weighted_scores, left_k_sum_scores])
            
            weighted_e_ij, weighted_scores, log_ksum_score = \
                outside_cache.gather(child_ids[:, :, 1].flatten(), 
                                     [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum])
            weighted_e_ij = weighted_e_ij.view(*child_ids_shape[:-1], dim)  # (batch_size, comb_size, dim)
            log_ksum_score = log_ksum_score.view(*child_ids_shape[:-1])  # (batch_size, comb_size)
            weighted_scores = weighted_scores.view(*child_ids_shape[:-1])

            right_k_sum_scores = torch.stack([log_ksum_score, out_scores[:, :, 1]], dim=2)  # (batch_size, comb_size, 2)
            right_k_weights = F.softmax(right_k_sum_scores, dim=2)
            right_weighted_e_ij = right_k_weights[:, :, 0].unsqueeze(2) * weighted_e_ij + \
                                    right_k_weights[:, :, 1].unsqueeze(2) * out_ikj[:, :, 1, :]
            right_weighted_scores = right_k_weights[:, :, 0] * weighted_scores + \
                                    right_k_weights[:, :, 1] * out_scores[:, :, 1]

            # (batch_size, comb_size, dim)
            right_k_sum_scores = right_k_sum_scores.logsumexp(dim=2, keepdim=True)

            right_weighted_e_ij = right_weighted_e_ij.view(-1, dim)
            right_weighted_scores = right_weighted_scores.view(-1, 1)
            right_k_sum_scores = right_k_sum_scores.view(-1, 1)
            
            outside_cache.scatter(child_ids[:, :, 1].flatten().long(), 
                                  [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                                  [right_weighted_e_ij, right_weighted_scores, right_k_sum_scores])

        return outside_cache
    

    def forward(self, 
                input_ids,
                tgt_ids=None,
                masks=None,
                merge_trajectory=None,
                atom_spans:List[List[int]]=None,
                pairwise=False,
                recover_tree=False):
        seq_lens = torch.sum(masks, dim=1,
                             dtype=torch.int)  # (batch_size)
        seq_lens_np = seq_lens.to('cpu', non_blocking=True)
        merge_trajectory = merge_trajectory.to('cpu', non_blocking=True)
        batch_size = input_ids.shape[0]
        
        flatten_input_ids, flatten_tgt_ids, input_embedding = \
            self.initialize_embeddings(input_ids, seq_lens, tgt_ids)

        if tgt_ids is not None:
            tgt_ids_cpu = flatten_tgt_ids.to('cpu', non_blocking=True)
        ids_num = flatten_input_ids.shape[0]
        input_cache_ids = torch.arange(SPECIAL_TOKEN_NUM, 
                                       SPECIAL_TOKEN_NUM + ids_num).to(self.device)

        inside_cache = self.create_tensor_cache(seq_lens_np)
        outside_cache = None
        group_embedding = self.empty_cell.unsqueeze(0).repeat(batch_size // 2, 1)

        inside_cache.scatter(input_cache_ids, [self.e_ij_id], [input_embedding])
        tables = CPPChartTableManager(seq_lens_np.data.numpy(), self.window_size, merge_trajectory.data.numpy(),
                                      inside_cache.placeholder_num, inside_cache.detach_offset)
        target_cache_ids, cache_ids, detach_cache_ids = \
                tables.construct_inside_groups(self.device)
        root_ids = tables.root_ids
        
        for iter_i in range(self.iter_times):
            if outside_cache is not None:
                inside_cache = self.create_tensor_cache(seq_lens)
                outside_input_embedding = outside_cache.gather(input_cache_ids, [self.e_ij_id])[0]
                inside_cache.scatter(input_cache_ids, [self.e_ij_id], [outside_input_embedding])

            splits_orders = self.inside(self.inout_encoders[iter_i], inside_cache, 
                                        zip(target_cache_ids, cache_ids, detach_cache_ids),
                                        outside_cache=outside_cache,
                                        is_final_layer=iter_i == self.iter_times - 1)
            root_embedding = inside_cache.gather(root_ids, [self.e_ij_id])[0]
            
            if pairwise:
                root_embedding, group_embedding = \
                    pairwise_contextual_inside_outside(root_embedding, self.inout_encoders[iter_i].inside, 
                                                       self.inout_encoders[iter_i].outside, group_embedding)

            outside_cache = self.outside(self.inout_encoders[iter_i], batch_size, root_ids, root_embedding, inside_cache,
                                         zip(reversed(target_cache_ids), reversed(cache_ids), reversed(detach_cache_ids)))
        
        split_ids, cache_ids = tables.best_trees(splits_orders, atom_spans)  # split_ids: -1 for terminal nodes, -100 for paddin
        # self attention
        if tgt_ids is not None:
            rebuilt_tgt_ids = rebuild_tgt_ids(split_ids, tgt_ids_cpu, pairwise)
            rebuilt_tgt_ids = rebuilt_tgt_ids.to(self.device, non_blocking=True)

        if not pairwise:
            rebuild_cache_ids = cache_ids
        else:
            group_ids = [[2 * pair_i, 2 * pair_i + 1] for pair_i in range(input_ids.shape[0] // 2)]
            rebuild_cache_ids = rebuild_batch_indices_cpu(cache_ids, group_ids, self.device, 
                                                          align_batch=False)

        rebuild_cache_ids = rebuild_cache_ids.to(self.device, non_blocking=True)
        
        span_masks = rebuild_cache_ids == 0  # for nn.TransformerEncoder, true means not allow to attend
        outside_repr = outside_cache.gather(rebuild_cache_ids.flatten(), [self.e_ij_id])[0]
        outside_repr = outside_repr.view(*rebuild_cache_ids.shape, -1)  # (N, L, dim)
        # add cls
        # TODO: replace it with group embedding and observe the performances on downstream tasks.
        # cls_embeddings = self.cls_embedding.unsqueeze(0).repeat(outside_repr.shape[0], 1)
        # outside_repr = torch.cat([cls_embeddings.unsqueeze(1), outside_repr], dim=1)
        if pairwise:
            outside_repr = torch.cat([group_embedding.unsqueeze(1), outside_repr], dim=1)
            span_masks = torch.cat([torch.zeros(span_masks.shape[0], 1, device=self.device), span_masks], dim=1)

        outputs = self.span_self_attention(outside_repr, src_key_padding_mask=span_masks)
        # (N, L + 1, dim)
        cls_embeddings = outputs[:, 0, :]
        if tgt_ids is not None:
            if pairwise:
                logits =  self.classifier(self.cls_dense(outputs[:, 1:, :]))  # (N, L, C)
            else:
                logits =  self.classifier(self.cls_dense(outputs))
            loss = F.cross_entropy(logits.permute(0, 2, 1), rebuilt_tgt_ids, ignore_index=-1)
        else:
            loss = torch.zeros((1,), dtype=torch.float, device=self.device)

        results = {}
        # estimate cross entropy loss
        results['loss'] = loss

        if recover_tree:
            targets = torch.full([input_ids.shape[0], 1, input_ids.shape[-1] - 1], fill_value=-1,
                                 requires_grad=False, dtype=torch.long, device=self.device)
            span_masks = torch.full([input_ids.shape[0], 1, input_ids.shape[-1] - 1,
                                input_ids.shape[-1] - 1], fill_value=0,
                                requires_grad=False, dtype=torch.int,
                                device=self.device)  # (batch_size, K, L - 1, L - 1)
            
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
            
            results['trees'] = [trees, {"split_masks": span_masks, "split_points": targets}]
            
        results['root_embeddings'] = root_embedding
        results['tensor_cache'] = outside_cache
        results['contextualized_embeddings'] = outputs[:, 1:, :] if pairwise else outputs
        if pairwise:
            results['group_embeddings'] = group_embedding
        results['cls_embedding'] = cls_embeddings
        
        return results