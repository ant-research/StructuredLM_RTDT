# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

import math
from turtle import position
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import deepcopy
import numpy as np
from .r2d2_common import ROLE_LEFT, ROLE_RIGHT

ACTIVATION_POOL = ['relu', 'gelu']


def _get_activation_fn(activation):
    if activation in ACTIVATION_POOL:
        return getattr(F, activation)

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TreeEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_role_count, 
                 activation='gelu', batch_first=False, val_position=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self._val_position = val_position
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_role_count, d_model)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, pos_ids=None):
        """
        :param src: concatenation of task embeddings and representation for left and right.
                    src shape: (task_embeddings + left + right, batch_size, dim)
        :param src_mask:
        :param pos_ids:
        :return:
        """
        if len(pos_ids.shape) == 1:
            sz = src.shape[1]  # sz: batch_size
            pos_ids = pos_ids.unsqueeze(1).expand(-1, sz)  # (3, batch_size)
        position_embedding = self.position_embedding(pos_ids)
        if not self._val_position:
            src2 = self.self_attn(src + position_embedding, src + position_embedding, 
                                  src, attn_mask=src_mask)[0]
        else:
            src2 = self.self_attn(src + position_embedding, src + position_embedding, 
                                  src + position_embedding, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class InsideEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.const_size = config.const_size
        self.left_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.right_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        
        layer = TreeEncoderLayer(config.hidden_size,
                                 config.num_attention_heads,
                                 config.intermediate_size,
                                 max_role_count=config.max_role_embeddings,
                                 dropout=config.attention_probs_dropout_prob,
                                 activation='gelu',
                                 batch_first=True,
                                 val_position=True)
        self.norm = nn.InstanceNorm1d(config.hidden_size)
        self.layers = nn.ModuleList([layer] + [deepcopy(layer) for _ in range(config.encoder_num_hidden_layers - 1)])
        self._device = None
        self._pos_ids = None
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    @property
    def pos_ids(self):
        if self._pos_ids is None:
            self._pos_ids = torch.arange(2).to(self.device)
        return self._pos_ids 

    def forward(self, src):
        """
        :param pos_ids:
        :param src_mask:
        :param src: [batch_size, comb_size, 2, dim]
        :return:
        """
        dim = src.shape[-1]
        org_shape = src.shape  # (batch_size, comb_size, 2, dim)
        output = src.view(-1, 2, dim)
        for mod in self.layers:
            output = mod(output, pos_ids=self.pos_ids.unsqueeze(0))
        
        left_const = self.left_linear(output[:, 0, :])
        right_const = self.right_linear(output[:, 1, :])
        mat_scores = torch.einsum("bi,bi->b", left_const, right_const) / math.sqrt(self.const_size)
        mat_scores = mat_scores.view(*org_shape[:-2])  # (batch_size, comb_size)

        return mat_scores, self.norm(torch.sum(output, dim=1).view(*org_shape[:-2], dim))
    
class ContextualOutsideEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.const_mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                       nn.GELU(),
                                       nn.Linear(config.hidden_size, config.const_size),
                                       nn.Sigmoid())
        self.left_mat = nn.Parameter(torch.rand(config.const_size, config.const_size))
        self.right_mat = nn.Parameter(torch.rand(config.const_size, config.const_size))
        
        layer = TreeEncoderLayer(config.hidden_size,
                                 config.num_attention_heads,
                                 config.intermediate_size,
                                 max_role_count=config.max_role_embeddings,
                                 dropout=config.attention_probs_dropout_prob,
                                 activation='gelu',
                                 batch_first=True,
                                 val_position=True)
        self.layers = nn.ModuleList([layer] + [deepcopy(layer) for _ in range(config.encoder_num_hidden_layers - 1)])
        self._device = None
        self._dec_pos_ids = None
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    @property
    def dec_pos_ids(self):
        if self._dec_pos_ids is None:
            self._dec_pos_ids = torch.tensor([[0, 1], [0, 2]], device=self.device)
        return self._dec_pos_ids 
    
    def forward(self, parent_ij, parent_scores, child_ikj, child_scores):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :param child_scores: (batch_size, comb_size, 2, 1)
        :return: (batch_size, 2), (batch_size, 2, dim)
        """
        # p_l = parent_ij @ self.W_outside_r  # (batch_size, dim)
        # out_score_ik = torch.einsum('bd, bcd->bc', p_l, child_ikj[:, :, 1, :])  # (batch_size, comb_size)
        # out_score_ik = out_score_ik + parent_scores + child_scores[:, :, 1, 0] # (batch_size, comb_size)

        # p_r = parent_ij @ self.W_outside_l  # (batch_size, dim)
        # out_score_kj = torch.einsum('bd, bcd->bc', p_r, child_ikj[:, :, 0, :])  # (batch_size, comb_size)
        # out_score_kj = out_score_kj + parent_scores + child_scores[:, :, 0, 0] # (batch_size, comb_size)

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        parent_const = self.const_mlp(parent_ij)  # (batch_size, comb_size, const_dim)
        child_const = self.const_mlp(child_ikj)  # (batch_size, comb_size, 2, const_dim)

        left_score = torch.einsum('bni,ij,bnj->bn', parent_const, self.left_mat, child_const[:, :, 1, :])
        right_score = torch.einsum('bni,ij,bnj->bn', parent_const, self.right_mat, child_const[:, :, 0, :])
        child_scores = child_scores.squeeze(3)
        out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
        out_score_kj = right_score + parent_scores + child_scores[:, :, 0]

        comb_size = child_ikj.shape[1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 2, 1)  # (batch_size, comb_size, 2, dim)
        
        inputs = torch.stack([parent_ij_ext, child_ikj.flip([2])], dim=3)  # (batch_size, comb_size, 2, 2, dim)
        inputs = inputs.view(batch_size * comb_size * 2, 2, -1)

        # self.dec_pos_ids: (2, 2)
        dec_pos_ids = self.dec_pos_ids.repeat(batch_size * comb_size, 1)
        for mod in self.dec_layers:
            inputs = mod(inputs, pos_ids=dec_pos_ids)

        out_e_ij = self.outside_norm(inputs.sum(dim=1))

        return torch.stack([out_score_ik, out_score_kj], dim=2), \
               out_e_ij.view(batch_size, comb_size, 2, -1)  # (batch_size, comb_size, 2, dim)
    
class OutsideEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.parent_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size, config.const_size))
        self.left_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.right_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self.const_size = config.const_size
        
        layer = TreeEncoderLayer(config.hidden_size,
                                 config.num_attention_heads,
                                 config.intermediate_size,
                                 max_role_count=config.max_role_embeddings,
                                 dropout=config.attention_probs_dropout_prob,
                                 activation='gelu',
                                 batch_first=True,
                                 val_position=True)
        self.norm = nn.InstanceNorm1d(config.hidden_size)
        self.layers = nn.ModuleList([layer] + [deepcopy(layer) for _ in range(config.decoder_num_hidden_layers - 1)])
        self._device = None
        self._dec_pos_ids = None
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    @property
    def dec_pos_ids(self):
        if self._dec_pos_ids is None:
            self._dec_pos_ids = torch.tensor([[0, 1], [0, 2]], device=self.device)
        return self._dec_pos_ids 
    
    def forward(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :param child_scores: (batch_size, comb_size, 2)
        :return: (batch_size, 2), (batch_size, 2, dim)
        """
        # p_l = parent_ij @ self.W_outside_r  # (batch_size, dim)
        # out_score_ik = torch.einsum('bd, bcd->bc', p_l, child_ikj[:, :, 1, :])  # (batch_size, comb_size)
        # out_score_ik = out_score_ik + parent_scores + child_scores[:, :, 1, 0] # (batch_size, comb_size)

        # p_r = parent_ij @ self.W_outside_l  # (batch_size, dim)
        # out_score_kj = torch.einsum('bd, bcd->bc', p_r, child_ikj[:, :, 0, :])  # (batch_size, comb_size)
        # out_score_kj = out_score_kj + parent_scores + child_scores[:, :, 0, 0] # (batch_size, comb_size)

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        
        comb_size = child_ikj.shape[1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 2, 1)  # (batch_size, comb_size, 2, dim)
        
        inputs = torch.stack([parent_ij_ext, child_ikj.flip([2])], dim=3)  # (batch_size, comb_size, 2, 2, dim)
        inputs = inputs.view(batch_size * comb_size * 2, 2, -1)

        # self.dec_pos_ids: (2, 2)
        pos_ids = self.dec_pos_ids.repeat(batch_size * comb_size, 1)
        for mod in self.layers:
            inputs = mod(inputs, pos_ids=pos_ids)
        # inputs: (?, 2, dim)

        outside_scores = None
        
        if parent_scores is not None and child_scores is not None:
            parent_const_r = self.parent_linear(inputs[::2, 0, :])
            right_child_const = self.right_linear(inputs[::2, 1, :])
            parent_const_l = self.parent_linear(inputs[1::2, 0, :])
            left_child_const = self.left_linear(inputs[1::2, 1, :])
            
            # parent_const = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            # left_child_const = self.left_linear(child_ikj[:, :, 0, :])  # (batch_size, comb_size, 2, const_dim)
            # right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            
            left_score = torch.einsum('bi,bi->b', parent_const_r, right_child_const) / math.sqrt(self.const_size)
            right_score = torch.einsum('bi,bi->b', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            left_score = left_score.view(batch_size, comb_size)
            right_score = right_score.view(batch_size, comb_size)
            out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
            out_score_kj = right_score + parent_scores + child_scores[:, :, 0]
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)

        out_e_ij = self.norm(inputs.sum(dim=1))

        return outside_scores, \
               out_e_ij.view(batch_size, comb_size, 2, -1)  # (batch_size, comb_size, 2, dim)


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
        self._device = None
        self._mask_cache = []
        self._pos_ids_cache = []
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, src, src_mask=None, pos_ids=None):
        """
        :param pos_ids:
        :param src_mask:
        :param src:
        :return:
        """
        output = src
        task_count = src.shape[1] - 2
        if pos_ids is None:
            while task_count >= len(self._pos_ids_cache):
                self._pos_ids_cache.append(None)
            if self._pos_ids_cache[task_count] is None:
                pos_ids = torch.tensor([0] * task_count + [ROLE_LEFT, ROLE_RIGHT], dtype=torch.long,
                                       device=self.device)
                self._pos_ids_cache[task_count] = pos_ids
            pos_ids = self._pos_ids_cache[task_count]
        if src_mask is None:
            while task_count >= len(self._mask_cache):
                self._mask_cache.append(None)
            if self._mask_cache[task_count] is None:
                src_mask = [[float('-inf') for _ in range(task_count + 2)] for _ in range(task_count + 2)]
                for pos_i in range(task_count + 2):
                    if pos_i < task_count:
                        src_mask[pos_i][pos_i] = 0
                    src_mask[pos_i][-1] = 0
                    src_mask[pos_i][-2] = 0
                src_mask = torch.tensor(src_mask, dtype=torch.float, device=self.device)
                self._mask_cache[task_count] = src_mask
            src_mask = self._mask_cache[task_count]

        output = src.permute(1, 0, 2)
        for mod in self.layers:
            output = mod(output, src_mask, pos_ids)

        return output.permute(1, 0, 2)

class ContextEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation='gelu'):
        super().__init__()
        self.num_attention_heads = nhead
        self.attention_head_size = int(d_model / nhead)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def split_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x
    
    def multi_head_attention(self, hidden_states, target, attn_mask):
        """
        :param emb: (max_len, dim)
        attn_mask: (L, D)
        """
        q = self.Q(hidden_states)  # (L, dim)
        k = self.K(target)  # (L, D, dim)
        v = self.V(target)  # (L, D, dim)

        query_layer = self.split_heads(q)  # (L, nhead, dim)
        key_layer = self.split_heads(k)  # (L, D, nhead, dim)
        value_layer = self.split_heads(v)  # (L, D, nhead, dim)
        query_layer = query_layer.permute(1, 0, 2)  # # (nhead, L, dim)
        key_layer = key_layer.permute(2, 0, 1, 3)  # (nhead, L, D, dim)
        value_layer = value_layer.permute(2, 0, 1,3)  # (nhead, L, D, dim)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer.unsqueeze(-2), key_layer.transpose(-1, -2))  # (nhead, L, 1, D)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attn_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attn_mask: (N, L, D)
            attention_scores = attention_scores + attn_mask.unsqueeze(0).unsqueeze(-2)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (nhead, L, 1, D)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (nhead, L, 1, dim)
        context_layer = context_layer.squeeze(-2)  # (nhead, L, dim)
        context_layer = context_layer.permute(1, 0, 2).contiguous()  # (L, nhead, dim)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
        
    def forward(self, src, target, attn_mask):
        src2 = self.multi_head_attention(src, target, attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # save memory        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class UniLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = ContextEncoderLayer(config.hidden_size,
                                    config.num_attention_heads,
                                    config.intermediate_size,
                                    dropout=config.attention_probs_dropout_prob,
                                    activation='gelu')
        self.max_positions = config.max_positions
        if hasattr(config, "bidirectional_pos") and config.bidirectional_pos is not None:
            self.bidirectional_pos = config.bidirectional_pos
        else:
            self.bidirectional_pos = False
        if self.bidirectional_pos:
            self.position_embeddings = nn.Embedding(config.max_positions * 2 + 2, config.embedding_dim)
        else:
            self.position_embeddings = nn.Embedding(config.max_positions + 2, config.embedding_dim)
        self.layers = nn.ModuleList([layer] + [deepcopy(layer) for _ in range(config.encoder_num_hidden_layers - 1)])
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, input_ids: List[List[int]] = None,
                memory: torch.Tensor = None, 
                embeddings: nn.Embedding = None):
        # memory : (N, 2, dim)
        assert len(input_ids) == memory.shape[0]
        N = memory.shape[0]
        dim = memory.shape[-1]
        mem_len = memory.shape[1]
        memory = memory.view(-1, dim)  # (N * 2, dim)
        flatten_ids = []
        max_key_len = 0
        pos_ids_batch = []
        offset = N * 2
        mem_pos_ids = [mem_pos % mem_len for mem_pos in range(mem_len * N)]
        fwd_pos_ids = []
        bwd_pos_ids = []
        for ids in input_ids:
            pos_ids = [offset + _ for _ in range(len(ids))]
            offset += len(ids)
            flatten_ids.extend(ids)
            max_key_len = max(len(ids) + mem_len, max_key_len)
            pos_ids_batch.append(pos_ids)
            for id_pos in range(len(ids)):
                fwd_pos_ids.append(id_pos + mem_len)
                bwd_pos_ids.append(len(ids) - 1 - id_pos + mem_len + self.max_positions)

        total_len =  2 * N + len(flatten_ids)
        gather_indices = np.zeros((total_len, max_key_len))
        mask = torch.full((total_len, max_key_len), fill_value=-np.inf, device=self.device)

        offset = 2 * N
        return_indices = []
        for sent_i in range(N):
            key_pos = [sent_i * mem_len + mem_idx for mem_idx in range(mem_len)]
            key_pos.extend(pos_ids_batch[sent_i])  # (mem_idx, context)
            key_len = len(key_pos)
            if len(key_pos) < max_key_len:
                key_pos.extend([pos_ids_batch[sent_i][-1]] * (max_key_len - key_len))
            return_indices.append(key_pos[mem_len:])
            gather_indices[sent_i * mem_len: mem_len * (sent_i + 1), :] = key_pos
            mask[sent_i * mem_len: mem_len * (sent_i + 1), key_len:] = 1

            ids = input_ids[sent_i]
            gather_indices[offset: offset + len(ids), :] = key_pos
            mask[sent_i * mem_len: mem_len * (sent_i + 1), :key_len] = 0
            mask[offset: offset + len(ids), :key_len] = 0
            offset += len(ids)

        flatten_ids = torch.tensor(flatten_ids, device=self.device)
        if not self.bidirectional_pos:
            fwd_pos_ids = torch.tensor(fwd_pos_ids, device=self.device, dtype=torch.long)
            input_embedding = embeddings(flatten_ids) + self.position_embeddings(fwd_pos_ids)
        else:
            fwd_pos_ids = torch.tensor(fwd_pos_ids, device=self.device, dtype=torch.long)
            bwd_pos_ids = torch.tensor(bwd_pos_ids, device=self.device, dtype=torch.long)
            input_embedding = embeddings(flatten_ids) + self.position_embeddings(fwd_pos_ids) + \
                self.position_embeddings(bwd_pos_ids)
        mem_pos_ids = torch.tensor(mem_pos_ids, device=self.device, dtype=torch.long)
        memory = memory + self.position_embeddings(mem_pos_ids)
        input = torch.cat([memory, input_embedding], dim=0)  # (total_len, dim)

        gather_indices = torch.tensor(gather_indices, device=self.device, dtype=torch.long)
        for layer in self.layers:
            tgt = input[gather_indices]  # (total_len, max_key_len, dim)
            input = layer(input, tgt, mask)

        return_indices = torch.tensor(return_indices, device=self.device, dtype=torch.long)
        return input[return_indices]
    
class PoolingLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size

        self.norm1 = nn.InstanceNorm1d(self.input_size)
        self.norm2 = nn.InstanceNorm1d(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))

        self.head_num = config.num_attention_heads

        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                               dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.position_embedding = nn.Embedding(3, config.hidden_size)
        
        self._device = None
        self._pos_ids = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def pos_ids(self):
        if self._pos_ids is None:
            self._pos_ids = torch.arange(3).to(self.device)
        return self._pos_ids

    def forward(self, children, parent):
        """
        :param children: (batch_size, 2, dim)
        :param parent: (batch_size, dim)
        :return:
        """
        dim = children.shape[-1]
        parent_ext = parent.unsqueeze(1)
        # (batch_size, group, 1, dim)
        context = torch.cat([parent_ext, children], dim=1)  # (batch_size, 3, dim)
        context_ = context.view(-1, 3, dim)
        role_embedding = self.position_embedding(self.pos_ids)  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src = parent_ext
        src2 = self.self_attn(src + role_embedding[:, :1, :], 
                              context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 1, dim)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src