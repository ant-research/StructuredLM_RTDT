import math
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
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
        src2 = self.self_attn(src + position_embedding, src + position_embedding, src, 
                              attn_mask=src_mask)[0]
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
        # self.const_linear = nn.Sequential(GroupLinear(2, config.hidden_size, config.hidden_size),
        #                                   nn.GELU(),
        #                                   GroupLinear(2, config.hidden_size, config.hidden_size))
        
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

    def forward(self, src, span_embeds=None):
        """
        :param src: [batch_size, comb_size, 2, dim]
        :param span_embeds: [batch_size, dim]
        :return:
        """
        dim = src.shape[-1]
        org_shape = src.shape  # (batch_size, comb_size, 2, dim)
        output = src.view(-1, 2, dim)

        # torch.cuda.synchronize()
        # with torch.cuda.stream(self.s1):
        left_const = self.left_linear(output[:, 0, :])
        right_const = self.right_linear(output[:, 1, :])
        
        for mod in self.layers:
            output = mod(output, pos_ids=self.pos_ids.unsqueeze(0))

        mat_scores = torch.einsum("bi,bi->b", left_const, right_const) / math.sqrt(self.const_size)
        mat_scores = mat_scores.view(*org_shape[:-2])  # (batch_size, comb_size)
        if span_embeds is not None:
            output = output.sum(dim=1).view(*org_shape[:-2], dim) + span_embeds.unsqueeze(1)
        else:
            output = output.sum(dim=1).view(*org_shape[:-2], dim)

        return mat_scores, self.norm(output)

    
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
        # self.const_linear = nn.Sequential(GroupLinear(3, config.hidden_size, config.hidden_size),
        #                                   nn.GELU(),
        #                                   GroupLinear(3, config.hidden_size, config.hidden_size))
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


        outside_scores = None
        
        if parent_scores is not None and child_scores is not None:
            parent_const = self.parent_linear(inputs[:, 0, :])
            right_child_const = self.right_linear(inputs[::2, 1, :])
            left_child_const = self.left_linear(inputs[1::2, 1, :])


        # self.dec_pos_ids: (2, 2)
        pos_ids = self.dec_pos_ids.repeat(batch_size * comb_size, 1)
        for mod in self.layers:
            inputs = mod(inputs, pos_ids=pos_ids)

        if parent_scores is not None and child_scores is not None:
            parent_const_r = parent_const[::2, :]
            parent_const_l = parent_const[1::2, :]
            left_score = (parent_const_r * right_child_const).sum(dim=-1) / math.sqrt(self.const_size)
            # right_score = torch.einsum('bi,bi->b', parent_const_l, left_child_const) / math.sqrt(self.const_size)
            right_score = (parent_const_l * left_child_const).sum(dim=-1) / math.sqrt(self.const_size)
            left_score = left_score.view(batch_size, comb_size)
            right_score = right_score.view(batch_size, comb_size)
            out_score_ik = left_score
            out_score_kj = right_score
            outside_scores = torch.stack([out_score_ik, out_score_kj], dim=2)
        # inputs: (?, 2, dim)
        out_e_ij = self.norm(inputs.sum(dim=1))

        return outside_scores, \
               out_e_ij.view(batch_size, comb_size, 2, -1)  # (batch_size, comb_size, 2, dim)