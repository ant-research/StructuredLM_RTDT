# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
from typing import List
import torch.nn as nn
from typing import Optional
from model.r2d2_base import R2D2Base
from data_structure.py_backend import CPPChartTableManager
from model.fast_parser import TransformerParser
from model.tree_encoder import InsideEncoder, OutsideEncoder
import torch
from torch.utils.checkpoint import checkpoint
from data_structure.tensor_cache import TensorCache, CacheType
from model.r2d2_common import SPECIAL_TOKEN_NUM
import torch.nn.functional as F
from datetime import datetime
from model.weighted_sum_func import WeightedSumFunc
from utils.math_util import gumbel_softmax
from dataclasses import dataclass


@dataclass
class InsideOutsideContext:
    scores: Optional = None, 
    attention_mask: Optional = None, 
    split_masks: Optional = None, 
    split_points: Optional = None,
    batch_size: Optional = None, 
    root_ids: Optional = None, 
    inside_cache: Optional = None, 
    outside_groups: Optional = None,
    input_cache_ids: Optional = None

DEFAULT_HEIGHT_THRESHOLD=15


class InsideOutsideModule(R2D2Base):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.parser_chunked = config.parser_chunked

        self.parser = TransformerParser(config)

        self.inside_enc = InsideEncoder(config)
        self.outside_enc = OutsideEncoder(config)
        self.outside_root_embedding = nn.Parameter(torch.rand(config.hidden_size))
        self.norm = nn.InstanceNorm1d(config.hidden_size)

        if config.ext_vocab_size > 0:
            self.ext_embeds = nn.Embedding(config.ext_vocab_size + 1, self.input_dim, padding_idx=0)
            # initialize with zero
            self.ext_embeds.weight.data.fill_(0.0)

        self.height_threshold = DEFAULT_HEIGHT_THRESHOLD
        if hasattr(config, 'height_threshold'):
            self.height_threshold = config.height_threshold
        self.use_gumbel = False
        if hasattr(config, 'use_gumbel'):
            self.use_gumbel = config.use_gumbel
        self.ldr_detach = False
        if hasattr(config, 'ldr_detach'):
            self.ldr_detach = config.ldr_detach

        self.e_ij_id = -1
        self.score_sum_id = -1
        self.score_ijk = -1
        self.height_ij = -1

        self.reduce_id = config.reduce_token_id
            
    def create_tensor_cache(self, seq_lens, total_cache_size=-1):
        # e_ij, log_p_ij, log_p_sum_ij
        tensor_cache = TensorCache(
            self.window_size,
            seq_lens,
            cache_types=[
                CacheType.NORMAL, CacheType.DETACH, 
                CacheType.NORMAL, CacheType.NORMAL
            ],
            dims=[self.input_dim, 1, 1, 1],
            placeholder_num=SPECIAL_TOKEN_NUM,
            device=self.device,
            total_cache_size=total_cache_size)
        self.e_ij_id = 0
        self.score_sum_id = 1
        self.score_ijk = 2
        self.height_ij = 3
        tensor_cache.fill(0, tensor_cache.capacity, [self.height_ij], [0])
        return tensor_cache
    
    def _flatten_inputs(self, input_ids, seq_lens, r2d2_embeddings):
        # Initialize embeddings
        block_size = input_ids.shape[-1]
        indices_gather = []
        for seq_i, seq_len in enumerate(seq_lens):
            indices_gather.extend(
                range(block_size * seq_i, block_size * seq_i + seq_len))

        flatten_input_ids = input_ids.flatten()
        indices_gather = torch.tensor(indices_gather, device=self.device)
        flatten_input_ids = flatten_input_ids.gather(
            dim=0, index=indices_gather)
        flatten_r2d2_emb = r2d2_embeddings.view(-1, r2d2_embeddings.shape[-1]).gather(
            dim=0, index=indices_gather.unsqueeze(1).repeat(1, r2d2_embeddings.shape[-1])
        )
            
        return flatten_input_ids, flatten_r2d2_emb

    def prepare_composition(self, group_ids, log_p_ids, tensor_cache):
        e_ij, h_ij = tensor_cache.gather(group_ids.flatten(), [self.e_ij_id, self.height_ij])
        log_p_ij = tensor_cache.gather(log_p_ids.flatten(), [self.score_sum_id])[0]
        e_ij = e_ij.view(*group_ids.shape, self.input_dim)
        h_ij = h_ij.view(*group_ids.shape)  # (batch_size, group_size, 2)
        log_p_ij = log_p_ij.view(*group_ids.shape) # (batch_size, group_size, 2)

        return e_ij, log_p_ij.sum(dim=-1), h_ij

    def inside(self,
               inside_cache,
               span_embeds,
               temperature,
               inside_groups):
        score_orders = []
        # a_ij_orders = []
        
        prepare_time = None
        inside_time = None
        weighted_time = None
        arg_sort_time = None
        for target_cache_ids, span_ids, cache_ids, detach_cache_ids in inside_groups:
            # target_cache_ids: (?)
            # cache_ids: (?, group_size, 2)
            # detach_cache_ids: (?, group_size, 2)

            # if candidate e_ij and log_p is not empty, apply composition function
            e_ij, scores_ij_sum, h_ij = self.prepare_composition(
                cache_ids, detach_cache_ids, inside_cache)
            # # e_ij: (batch_size, group_size, 2, dim), c_ij: (batch_size, 2, dim)
            
            if span_embeds is None:
                scores_ijk, c_ijk = self.inside_enc(e_ij)
            else:
                scores_ijk, c_ijk = self.inside_enc(e_ij, span_embeds[span_ids, :])
            # scores_ijk, c_ijk = checkpoint(self.inside_enc, e_ij, use_reentrant=False)

            # expected output put c_ijk: (batch_size, group_size, dim)
            # log_p_ijk: (batch_size, group_size)
            # print(scores_ijk.shape)
            # print(scores_ij_sum.shape)
            scores_ijk_sum = scores_ijk  # (batch_size, combination_size)

            # assert not torch.any(torch.isinf(log_p_ij_step))
            if not self.use_gumbel:
                a_ij = F.softmax(scores_ijk_sum / temperature, dim=-1)
            else:
                a_ij = gumbel_softmax(scores_ijk_sum, temperature)

            # (batch_size, combination_size)
            
            # c_ij = torch.einsum("ij,ijk->ik", a_ij, c_ijk)
            c_ij = WeightedSumFunc.apply(a_ij, c_ijk)
            c_ij = self.norm(c_ij)
            
            # c_ij_detach = torch.einsum("ij,ijk->ik", a_ij.detach(), c_ijk)
            h_ij_next, _ = h_ij.max(dim=-1)  # (batch_size, group_size)
            h_ij_next = h_ij_next + 1
            h_ij = torch.einsum("ij, ij->i", a_ij, h_ij_next)  # (batch_size)
            
            scores_ij_sum = torch.einsum("ij, ij->i", a_ij, scores_ijk_sum).unsqueeze(1)

            inside_cache.scatter(target_cache_ids, [self.e_ij_id, self.score_sum_id, self.height_ij],
                                [c_ij, scores_ij_sum, h_ij.unsqueeze(1)])
            
            # padding to group_size
            score_orders.append(scores_ijk_sum.argsort(dim=1, descending=True).to('cpu', non_blocking=True))
            # a_ij_orders.append(a_ij.argsort(dim=1, descending=True).to('cpu', non_blocking=True))

        return score_orders #, a_ij_orders

    def outside_embeddings(self, ctx):
        root_embedding = self.outside_root_embedding.unsqueeze(0).repeat(ctx.batch_size, 1)
        outside_cache = self.outside(ctx.batch_size, ctx.root_ids, root_embedding, \
                                     ctx.inside_cache, ctx.outside_groups)
        outside_repr = outside_cache.gather(ctx.input_cache_ids, [self.e_ij_id])[0]
        return outside_repr
        
    def parser_loss(self, ctx):
        # split_masks: (batch_size, L - 1, L - 1)
        # split points: (batch_size, L - 1)
        scores = ctx.scores
        split_masks = ctx.split_masks
        split_points = ctx.split_points
        L = scores.shape[1]
        attention_mask = ctx.attention_mask

        assert len(attention_mask.shape) == 2
        scores.masked_fill_(attention_mask[:, 1: L + 1] == 0, float('-inf'))
        scores = scores.unsqueeze(1).repeat(1, L, 1)
        scores.masked_fill_(split_masks[:, :L, :L] == 0, float('-inf'))  # (batch_size, L - 1, L - 1)

        # test only feedback on root split
        # log_p = F.log_softmax(scores.float(), dim=-1)  # (batch_size, L - 1, L - 1)
        return F.cross_entropy(scores.transpose(1, 2).float(), split_points[:, :L], ignore_index=-1)  

    def outside(self, batch_size, root_ids, root_embedding, inside_cache, outside_groups):
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
        for target_cache_ids, cache_ids, detach_cache_ids in outside_groups:
            parent_ids = target_cache_ids
            child_ids = cache_ids # (N, comb_size, 2)
            
            # assert child_ids[:, :, 0].unique().shape[0] == cache_ids.shape[0] * cache_ids.shape[1]
            # assert child_ids[:, :, 1].unique().shape[0] == cache_ids.shape[0] * cache_ids.shape[1]

            score_ids = detach_cache_ids
            
            parent_ij, parent_ij_score = outside_cache.gather(parent_ids, [topdown_e_ij_slot, topdown_score_slot])

            child_ids_shape = child_ids.shape  # (batch_size, comb_size, 2)
            child_ikj = inside_cache.gather(child_ids.flatten(), [self.e_ij_id])[0]
            child_scores = inside_cache.gather(score_ids.flatten(), [self.score_sum_id])[0]
            child_ikj = child_ikj.view(*child_ids.shape, -1)
            child_scores = child_scores.view(*child_ids.shape)  # (batch_size, comb_size, 2)

            out_scores, out_ikj = self.outside_enc(parent_ij, child_ikj, parent_ij_score, child_scores)
            # out_scores, out_ikj = checkpoint(self.outside_enc, parent_ij, child_ikj, parent_ij_score, child_scores, use_reentrant=False)
            # span_norm = (1 + max_lens - span_lens).unsqueeze(1).unsqueeze(2)
            # out_ikj: (batch_size, comb_size, 2, dim)
            
            dim = out_ikj.shape[-1]

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
                chunk_input_ids,
                chunk_masks,
                input_ids,
                masks,
                r2d2_embeddings, # corresponding to chunked_input_ids
                group_ids,
                max_input_len,
                atom_spans:List[List[int]]=None,
                eos_labels=None,
                span_ids=None,
                external_vocab_ids=None,
                coeff=1.0,
                temperature=1.0):

        split_indices, split_scores = self.parser(chunk_input_ids, chunk_masks, atom_spans=atom_spans, noise_coeff=coeff)
        split_indices = split_indices.to('cpu', non_blocking=True)

        seq_lens = torch.sum(masks, dim=1, dtype=torch.int)  # (batch_size)
        seq_lens_np = seq_lens.to('cpu').data.numpy()
        
        if len(chunk_masks.shape) == 2:
            chunk_seq_lens_np = (chunk_masks != 0).sum(dim=1).cpu().data.numpy()
            # chunk_seq_lens_np = chunk_masks.sum(dim=1).cpu().data.numpy()
        # elif len(chunk_masks.shape) == 3:
        #     chunk_seq_lens_np = (chunk_masks.sum(dim=1) > 0).cpu().to(int).sum(dim=1).data.numpy()

        batch_size = input_ids.shape[0]
        input_ids_cpu = input_ids.to('cpu', non_blocking=True)
        
        flatten_input_ids, flatten_r2d2_emb = self._flatten_inputs(chunk_input_ids, chunk_seq_lens_np, r2d2_embeddings)
        ids_num = flatten_input_ids.shape[0]
        input_cache_ids = torch.arange(SPECIAL_TOKEN_NUM, 
                                       SPECIAL_TOKEN_NUM + ids_num).to(self.device)
        
        inside_cache = self.create_tensor_cache(seq_lens_np)
        inside_cache.scatter(input_cache_ids, [self.e_ij_id], [flatten_r2d2_emb])

        tables = CPPChartTableManager(seq_lens_np, self.window_size, split_indices.data.numpy(),
                                      inside_cache.placeholder_num, inside_cache.detach_offset, group_ids=group_ids,
                                      span_ids=span_ids)
        target_cache_ids, span_ids, cache_ids, detach_cache_ids = \
                tables.construct_inside_groups(self.device)
        root_ids = tables.root_ids

        span_embeds = None
        if external_vocab_ids is not None:
            span_embeds = self.ext_embeds(external_vocab_ids)

        score_orders = self.inside(inside_cache, span_embeds, temperature,
                                    zip(target_cache_ids, span_ids, cache_ids, detach_cache_ids))
        
        span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, token_indices, ext_ids = \
            tables.prepare_generation(score_orders, score_orders, atom_spans, input_ids_cpu.data.numpy(),
                                      group_ids, self.eos_token_id, self.reduce_id,
                                      max_input_len, eos_labels=eos_labels)
        # span_mask, split_targets, ldr_cache_ids, position_ids, tgt_ids

        ldr_cache_ids = ldr_cache_ids.to(self.device, non_blocking=True)
        position_ids = position_ids.to(self.device, non_blocking=True)
        tgt_ids = tgt_ids.to(self.device, non_blocking=True)
        ext_ids = ext_ids.to(self.device, non_blocking=True)
        token_indices = token_indices.to(self.device, non_blocking=True)
        
        span_masks = span_masks.to(self.device, non_blocking=True)
        split_targets = split_targets.to(self.device, non_blocking=True)

        ldr_repr = inside_cache.gather(ldr_cache_ids.flatten(), [self.e_ij_id])[0]
        ldr_repr = ldr_repr.view(*ldr_cache_ids.shape, -1)  # (N, L, dim)

        # l_height = (inside_cache.gather(root_ids, [self.height_ij])[0] / seq_lens).mean()
        inside_height = inside_cache.gather(root_ids, [self.height_ij])[0]

        inside_height = torch.where(inside_height > self.height_threshold, inside_height - self.height_threshold, 0)
        height_norm = torch.where(seq_lens > self.height_threshold, seq_lens - self.height_threshold, 1)
        l_height = (inside_height / height_norm).mean()

        ctx = InsideOutsideContext(
            scores=split_scores, 
            attention_mask=chunk_masks,
            split_masks=span_masks, 
            split_points=split_targets,
            batch_size=batch_size, 
            root_ids=root_ids, 
            inside_cache=inside_cache, 
            input_cache_ids=input_cache_ids,
            outside_groups=zip(reversed(target_cache_ids), reversed(cache_ids), reversed(detach_cache_ids))
        )
        if self.ldr_detach:
            ldr_repr = ldr_repr.detach()
        
        return ctx, flatten_input_ids, ldr_repr, position_ids, \
            tgt_ids, token_indices, ext_ids, split_targets, l_height