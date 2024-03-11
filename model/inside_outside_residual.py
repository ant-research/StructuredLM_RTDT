# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

import math
import torch.nn as nn
import torch
from collections import namedtuple

from model.tree_encoder import _get_activation_fn


ContextualizedCells = namedtuple("ContextualizedCells", [
    "e_ij_indices", "ctx_cells_k", "ctx_cells_v", "ctx_mask", "ctx_scores"])


class InsideLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size

        self.cmp_norm1 = nn.LayerNorm(self.input_size)
        self.cmp_norm2 = nn.LayerNorm(self.input_size)
        self.cmp_dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.cmp_dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.cmp_mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                     nn.GELU(),
                                     nn.Dropout(config.attention_probs_dropout_prob),
                                     nn.Linear(config.intermediate_size, self.input_size))

        self.topk = config.topk
        self.head_num = config.num_attention_heads

        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                               dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.position_embedding = nn.Embedding(3, config.hidden_size)
        
        self.const_size = config.const_size
        self.left_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.right_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self._pos_ids = None
        
        self.norm = nn.LayerNorm(self.input_size)
        
        self._device = None
        self._pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.position_embedding.weight.data.normal_(mean=0, std=0.02)

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
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim)
        :return:
        """
        batch_size = parent.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
        # (batch_size, group, 1, dim)
        context = torch.cat([parent_ext, children], dim=2)  # (batch_size, group, 3, dim)
        src = parent_ext.view(-1, 1, dim)  # (?, 1, dim)
        context_ = context.view(-1, 3, dim)

        role_embedding = self.position_embedding(self.pos_ids)  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src2 = self.self_attn(src + role_embedding[:, :1, :], 
                              context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 1, dim)
        src = src + self.cmp_dropout1(src2)
        src = self.cmp_norm1(src)
        src2 = self.cmp_mlp(src)
        src = src + self.cmp_dropout2(src2)
        src = self.cmp_norm2(src)

        src = src.view(batch_size, group_size, dim)
        
        left_const = self.left_linear(children[:, :, 0, :])
        right_const = self.right_linear(children[:, :, 1, :])
        mat_scores = torch.einsum("bgi,bgi->bg", left_const, right_const) / math.sqrt(self.const_size)
        
        return mat_scores, src
    
class InsideShareMLPLayer(nn.Module):
    def __init__(self, config, inside_fn) -> None:
        super().__init__()
        self.input_size = config.hidden_size

        self.cmp_norm1 = nn.LayerNorm(self.input_size)
        self.cmp_norm2 = nn.LayerNorm(self.input_size)
        self.cmp_dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.cmp_dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.cmp_mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                     nn.GELU(),
                                     nn.Dropout(config.attention_probs_dropout_prob),
                                     nn.Linear(config.intermediate_size, self.input_size))

        self.topk = config.topk
        self.head_num = config.num_attention_heads

        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                               dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.position_embedding = nn.Embedding(3, config.hidden_size)
        
        self.inside_fn = inside_fn
        self._pos_ids = None
        
        self.norm = nn.LayerNorm(self.input_size)
        
        self._device = None
        self._pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.position_embedding.weight.data.normal_(mean=0, std=0.02)

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
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim)
        :return:
        """
        batch_size = parent.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
        # (batch_size, group, 1, dim)
        context = torch.cat([parent_ext, children], dim=2)  # (batch_size, group, 3, dim)
        src = parent_ext.view(-1, 1, dim)  # (?, 1, dim)
        context_ = context.view(-1, 3, dim)

        role_embedding = self.position_embedding(self.pos_ids)  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src2 = self.self_attn(src + role_embedding[:, :1, :], 
                              context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 1, dim)
        src = src + self.cmp_dropout1(src2)
        src = self.cmp_norm1(src)
        src2 = self.cmp_mlp(src)
        src = src + self.cmp_dropout2(src2)
        src = self.cmp_norm2(src)

        src = src.view(batch_size, group_size, dim)
        
        mat_scores = self.inside_fn(children)
        
        return mat_scores, src


class OutsideLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

        self.left_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.right_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self.parent_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size, config.const_size))

        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                               dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.position_embedding = nn.Embedding(3, config.hidden_size)
        
        self._device = None
        self._dec_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.position_embedding.weight.data.normal_(mean=0, std=0.02)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def dec_pos_ids(self):
        if self._dec_pos_ids is None:
            self._dec_pos_ids = torch.tensor([0, 1, 2], device=self.device)
        return self._dec_pos_ids 

    def forward(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        src = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.position_embedding(self.dec_pos_ids).unsqueeze(0)
        src2 = self.self_attn(src + role_embedding[:, :2, :], context + role_embedding, 
                              context + role_embedding)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, comb_size, 2, dim)
        
        if parent_scores is not None and child_scores is not None:
            parent_const_r = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            parent_const_l = self.parent_linear(parent_ij)
            left_child_const = self.left_linear(child_ikj[:, :, 0, :])
            
            # parent_const = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            # left_child_const = self.left_linear(child_ikj[:, :, 0, :])  # (batch_size, comb_size, 2, const_dim)
            # right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            
            left_score = torch.einsum('bi,bgi->bg', parent_const_r, right_child_const) / math.sqrt(self.const_size)
            right_score = torch.einsum('bi,bgi->bg', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
            out_score_kj = right_score + parent_scores + child_scores[:, :, 0]
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)
        else:
            outside_scores = None

        return outside_scores, src
    
class OutsideShareMLPLayer(nn.Module):
    def __init__(self, config, outside_fn) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

        self.outside_fn = outside_fn

        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                               dropout=config.attention_probs_dropout_prob, batch_first=True)
        self.position_embedding = nn.Embedding(3, config.hidden_size)
        
        self._device = None
        self._dec_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.position_embedding.weight.data.normal_(mean=0, std=0.02)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def dec_pos_ids(self):
        if self._dec_pos_ids is None:
            self._dec_pos_ids = torch.tensor([0, 1, 2], device=self.device)
        return self._dec_pos_ids 

    def forward(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        src = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.position_embedding(self.dec_pos_ids).unsqueeze(0)
        src2 = self.self_attn(src + role_embedding[:, :2, :], context + role_embedding, 
                              context + role_embedding)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, comb_size, 2, dim)
        
        if parent_scores is not None and child_scores is not None:
            outside_scores = self.outside_fn(parent_ij, child_ikj, parent_scores, child_scores)
        else:
            outside_scores = None

        return outside_scores, src

class TreeEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, 
                 activation='gelu', batch_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, role_embeddings=None):
        """
        :param src: concatenation of task embeddings and representation for left and right.
                    src shape: (task_embeddings + left + right, batch_size, dim)
        :param src_mask:
        :param pos_ids:
        :return:
        """
        if role_embeddings is not None:
            src2 = self.self_attn(src + role_embeddings, src + role_embeddings, 
                                  src + role_embeddings)[0]
        else:
            src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    
class InsideOutsideMultiLayerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size

        # inside parameters
        self.inside_roles = nn.Embedding(3, config.hidden_size)
        
        self.inside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.inside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        
        self.inside_norm = nn.LayerNorm(self.input_size)
        
        # outside parameters
        self.outside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self.outside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size, config.const_size))
        self.outside_parent = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.GELU(),
                                            nn.Linear(config.hidden_size, config.const_size))

        self.outside_roles = nn.Embedding(3, config.hidden_size)
        

        self.inside_layers = nn.ModuleList(
            [TreeEncoderLayer(config.hidden_size,
                              config.num_attention_heads,
                              config.intermediate_size,
                              dropout=config.attention_probs_dropout_prob,
                              activation='gelu',
                              batch_first=True) for _ in range(config.encoder_num_hidden_layers)])

        if config.share_inside_outside:
            self.outside_layers = self.inside_layers
        else:
            self.outside_layers = nn.ModuleList(
                [TreeEncoderLayer(config.hidden_size,
                                  config.num_attention_heads,
                                  config.intermediate_size,
                                  dropout=config.attention_probs_dropout_prob,
                                  activation='gelu',
                                  batch_first=True) for _ in range(config.decoder_num_hidden_layers)])
        
        self._device = None
        self._inside_pos_ids = None
        self._outside_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.inside_roles.weight.data.normal_(mean=0, std=0.02)
        self.outside_roles.weight.data.normal_(mean=0, std=0.02)
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def inside_pos_ids(self):
        if self._inside_pos_ids is None:
            self._inside_pos_ids = torch.arange(2).to(self.device)
        return self._inside_pos_ids

    @property
    def outside_pos_ids(self):
        if self._outside_pos_ids is None:
            self._outside_pos_ids = torch.arange(3).to(self.device)
        return self._outside_pos_ids
        
    def inside(self, children, parents=None):
        """
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim) or (batch_size, group, dim)
        :return: (batch_size, group, dim)
        """
        batch_size = children.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        
        output = children.view(batch_size * group_size, 2, dim)
        role_embeddings = self.inside_roles(self.inside_pos_ids)  # (2, dim)
        output = output + role_embeddings.unsqueeze(0)
        for inside_layer in self.inside_layers:
            output = inside_layer(output)
        
        output = output.view(batch_size, group_size, 2, dim)
        left_const = self.inside_left(output[:, :, 0, :])
        right_const = self.inside_right(output[:, :, 1, :])
        mat_scores = torch.einsum("bgi,bgi->bg", left_const, right_const) / math.sqrt(self.const_size)
        src = self.inside_norm(output.sum(dim=2))  # (batch_size, group_size, dim)
        
        return mat_scores, src
    
    def outside(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        # output = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.outside_roles(self.outside_pos_ids).unsqueeze(0)
        output = context + role_embedding
        for outside_layer in self.outside_layers:
            output = outside_layer(output)
        
        # output: (batch_size, comb_size, 3, dim)
        output = output.view(batch_size, comb_size, 3, dim)
        
        if parent_scores is not None and child_scores is not None:
            parent_const_r = self.outside_parent(parent_ij)  # (batch_size, const_dim)
            right_child_const = self.outside_right(child_ikj[:, :, 1, :])
            parent_const_l = self.outside_parent(parent_ij)
            left_child_const = self.outside_left(child_ikj[:, :, 0, :])
            
            # parent_const = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            # left_child_const = self.left_linear(child_ikj[:, :, 0, :])  # (batch_size, comb_size, 2, const_dim)
            # right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            
            left_score = torch.einsum('bi,bgi->bg', parent_const_r, right_child_const) / math.sqrt(self.const_size)
            right_score = torch.einsum('bi,bgi->bg', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
            out_score_kj = right_score + parent_scores + child_scores[:, :, 0]
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)
        else:
            outside_scores = None

        return outside_scores, output[:, :, :2, :].contiguous()

class InsideOutsideShareMLPEncoder(nn.Module):
    def __init__(self, config, inside_fn, outside_fn) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size

        # inside parameters
        self.inside_roles = nn.Embedding(3, config.hidden_size)

        self.outside_roles = nn.Embedding(3, config.hidden_size)
        
        self.inside_fn = inside_fn
        self.outside_fn = outside_fn
        
        # shared params
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))

        self.self_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                                    dropout=config.attention_probs_dropout_prob, batch_first=True)
        
        self._device = None
        self._inside_pos_ids = None
        self._outside_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.inside_roles.weight.data.normal_(mean=0, std=0.02)
        self.outside_roles.weight.data.normal_(mean=0, std=0.02)
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def inside_pos_ids(self):
        if self._inside_pos_ids is None:
            self._inside_pos_ids = torch.arange(3).to(self.device)
        return self._inside_pos_ids

    @property
    def outside_pos_ids(self):
        if self._outside_pos_ids is None:
            self._outside_pos_ids = torch.arange(3).to(self.device)
        return self._outside_pos_ids
        
    def inside(self, children, parent):
        """
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim) or (batch_size, group, dim)
        :return:
        """
        batch_size = parent.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        if len(parent.shape) == 2:
            parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
        else:
            parent_ext = parent.unsqueeze(2)
        # (batch_size, group, 1, dim)
        context = torch.cat([parent_ext, children], dim=2)  # (batch_size, group, 3, dim)
        src = parent_ext.view(-1, 1, dim)  # (?, 1, dim)
        context_ = context.view(-1, 3, dim)

        role_embedding = self.inside_roles(self.inside_pos_ids)  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src2 = self.self_attention(src + role_embedding[:, :1, :], 
                                   context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 1, dim)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, group_size, dim)
        
        # left_const = self.inside_left(children[:, :, 0, :])
        # right_const = self.inside_right(children[:, :, 1, :])
        # mat_scores = torch.einsum("bgi,bgi->bg", left_const, right_const) / math.sqrt(self.const_size)
        mat_scores = self.inside_fn(children)
        
        return mat_scores, src
    
    def outside(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        src = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.outside_roles(self.outside_pos_ids).unsqueeze(0)
        src2 = self.self_attention(src + role_embedding[:, :2, :], context + role_embedding, 
                                   context + role_embedding)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, comb_size, 2, dim)
        
        if parent_scores is not None and child_scores is not None:
            outside_scores = self.outside_fn(parent_ij, child_ikj, parent_scores, child_scores)
        else:
            outside_scores = None

        return outside_scores, src 
    
class InsideOutsideLayerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size

        # inside parameters
        self.inside_roles = nn.Embedding(3, config.hidden_size)
        
        self.inside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.inside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        
        self.inside_mlp_norm = nn.LayerNorm(self.input_size)
        
        # outside parameters
        self.outside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self.outside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size, config.const_size))
        self.outside_parent = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.GELU(),
                                            nn.Linear(config.hidden_size, config.const_size))

        self.outside_roles = nn.Embedding(3, config.hidden_size)
        
        # shared params
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))

        self.self_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                                    dropout=config.attention_probs_dropout_prob, batch_first=True)
        
        self._device = None
        self._inside_pos_ids = None
        self._outside_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.inside_roles.weight.data.normal_(mean=0, std=0.02)
        self.outside_roles.weight.data.normal_(mean=0, std=0.02)
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def inside_pos_ids(self):
        if self._inside_pos_ids is None:
            self._inside_pos_ids = torch.arange(3).to(self.device)
        return self._inside_pos_ids

    @property
    def outside_pos_ids(self):
        if self._outside_pos_ids is None:
            self._outside_pos_ids = torch.arange(3).to(self.device)
        return self._outside_pos_ids
        
    def inside(self, children, parent):
        """
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim) or (batch_size, group, dim)
        :return:
        """
        batch_size = parent.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        if len(parent.shape) == 2:
            parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
        else:
            parent_ext = parent.unsqueeze(2)
        # (batch_size, group, 1, dim)
        context = torch.cat([parent_ext, children], dim=2)  # (batch_size, group, 3, dim)
        src = parent_ext.view(-1, 1, dim)  # (?, 1, dim)
        context_ = context.view(-1, 3, dim)

        role_embedding = self.inside_roles(self.inside_pos_ids)  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src2 = self.self_attention(src + role_embedding[:, :1, :], 
                                   context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 1, dim)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, group_size, dim)
        
        left_const = self.inside_left(children[:, :, 0, :])
        right_const = self.inside_right(children[:, :, 1, :])
        mat_scores = torch.einsum("bgi,bgi->bg", left_const, right_const) / math.sqrt(self.const_size)
        
        return mat_scores, src
    
    def outside(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        src = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.outside_roles(self.outside_pos_ids).unsqueeze(0)
        src2 = self.self_attention(src + role_embedding[:, :2, :], context + role_embedding, 
                                   context + role_embedding)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, comb_size, 2, dim)
        
        if parent_scores is not None and child_scores is not None:
            parent_const_r = self.outside_parent(parent_ij)  # (batch_size, const_dim)
            right_child_const = self.outside_right(child_ikj[:, :, 1, :])
            parent_const_l = self.outside_parent(parent_ij)
            left_child_const = self.outside_left(child_ikj[:, :, 0, :])
            
            # parent_const = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            # left_child_const = self.left_linear(child_ikj[:, :, 0, :])  # (batch_size, comb_size, 2, const_dim)
            # right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            
            left_score = torch.einsum('bi,bgi->bg', parent_const_r, right_child_const) / math.sqrt(self.const_size)
            right_score = torch.einsum('bi,bgi->bg', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
            out_score_kj = right_score + parent_scores + child_scores[:, :, 0]
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)
        else:
            outside_scores = None

        return outside_scores, src
    
    
class InsideOutsideResLayerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.const_size = config.const_size

        # inside parameters
        self.inside_roles = nn.Embedding(3, config.hidden_size)
        
        self.inside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(config.hidden_size, config.const_size))
        self.inside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        
        self.inside_mlp_norm = nn.LayerNorm(self.input_size)
        
        # outside parameters
        self.outside_left = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.GELU(),
                                          nn.Linear(config.hidden_size, config.const_size))
        self.outside_right = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size, config.const_size))
        self.outside_parent = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.GELU(),
                                            nn.Linear(config.hidden_size, config.const_size))

        self.outside_roles = nn.Embedding(3, config.hidden_size)
        
        # shared params
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.mlp = nn.Sequential(nn.Linear(self.input_size, config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(config.attention_probs_dropout_prob),
                                 nn.Linear(config.intermediate_size, self.input_size))

        self.self_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, 
                                                    dropout=config.attention_probs_dropout_prob, batch_first=True)
        
        self._device = None
        self._inside_pos_ids = None
        self._outside_pos_ids = None
        self._init_weights()
        
    def _init_weights(self):
        self.inside_roles.weight.data.normal_(mean=0, std=0.02)
        self.outside_roles.weight.data.normal_(mean=0, std=0.02)
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    @property
    def inside_pos_ids(self):
        if self._inside_pos_ids is None:
            self._inside_pos_ids = torch.arange(3).to(self.device)
        return self._inside_pos_ids

    @property
    def outside_pos_ids(self):
        if self._outside_pos_ids is None:
            self._outside_pos_ids = torch.arange(3).to(self.device)
        return self._outside_pos_ids
        
    def inside(self, children, parent):
        """
        :param children: (batch_size, group, 2, dim)
        :param parent: (batch_size, dim) or (batch_size, group, dim)
        :return:
        """
        batch_size = children.shape[0]
        group_size = children.shape[1]
        dim = children.shape[-1]
        # if len(parent.shape) == 2:
        #     parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
        # else:
        #     parent_ext = parent.unsqueeze(2)
        if parent is None:
            context = children
            seq_len = 2
        else:
            parent_ext = parent.unsqueeze(1).repeat(1, group_size, 1).unsqueeze(2)
            context = torch.cat([children, parent_ext], dim=2)  # (batch_size, group, 3, dim)
            seq_len = 3
        # (batch_size, group, 1, dim)
        
        # src = parent_ext.view(-1, 1, dim)  # (?, 1, dim)
        context_ = context.view(-1, seq_len, dim)
        src = context_

        role_embedding = self.inside_roles(self.inside_pos_ids)[:seq_len, :]  # (3, dim)
        role_embedding = role_embedding.unsqueeze(0)
        src2 = self.self_attention(context_ + role_embedding, 
                                   context_ + role_embedding, context_ + role_embedding)[0]
        # (?, 3, dim)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src.sum(dim=1))

        src = src.view(batch_size, group_size, dim)
        
        left_const = self.inside_left(children[:, :, 0, :])
        right_const = self.inside_right(children[:, :, 1, :])
        mat_scores = torch.einsum("bgi,bgi->bg", left_const, right_const) / math.sqrt(self.const_size)
        
        return mat_scores, src
    
    def outside(self, parent_ij, child_ikj, parent_scores=None, child_scores=None):
        """
        :param parent_ij: (batch_size, dim)
        :param parent_scores: (batch_size, 1)
        :param child_ikj: (batch_size, comb_size, 2, dim)
        :return: (batch_size, comb_size, 2, dim)
        """

        batch_size = child_ikj.shape[0]
        comb_size = child_ikj.shape[1]
        dim = child_ikj.shape[-1]
        parent_ij_ext = parent_ij.unsqueeze(1).unsqueeze(2).repeat(1, comb_size, 1, 1)
        # (batch_size, comb_size, 1, dim)
        context = torch.cat([child_ikj, parent_ij_ext], dim=2)
        
        # (batch_size, comb_size, 3, dim)
        context = context.view(batch_size * comb_size, 3, dim)
        src = child_ikj.view(batch_size * comb_size, 2, dim)
        role_embedding = self.outside_roles(self.outside_pos_ids).unsqueeze(0)
        src2 = self.self_attention(src + role_embedding[:, :2, :], context + role_embedding, 
                                   context + role_embedding)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.view(batch_size, comb_size, 2, dim)
        
        if parent_scores is not None and child_scores is not None:
            parent_const_r = self.outside_parent(parent_ij)  # (batch_size, const_dim)
            right_child_const = self.outside_right(child_ikj[:, :, 1, :])
            parent_const_l = self.outside_parent(parent_ij)
            left_child_const = self.outside_left(child_ikj[:, :, 0, :])
            
            # parent_const = self.parent_linear(parent_ij)  # (batch_size, const_dim)
            # left_child_const = self.left_linear(child_ikj[:, :, 0, :])  # (batch_size, comb_size, 2, const_dim)
            # right_child_const = self.right_linear(child_ikj[:, :, 1, :])
            
            left_score = torch.einsum('bi,bgi->bg', parent_const_r, right_child_const) / math.sqrt(self.const_size)
            right_score = torch.einsum('bi,bgi->bg', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            out_score_ik = left_score + parent_scores + child_scores[:, :, 1]
            out_score_kj = right_score + parent_scores + child_scores[:, :, 0]
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)
        else:
            outside_scores = None

        return outside_scores, src