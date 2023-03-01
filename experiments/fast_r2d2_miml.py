# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu


from math import sqrt
import numpy as np
import torch.nn as nn
import torch
from model.r2d2_cuda import R2D2Cuda
from model.topdown_parser import LSTMParser
import torch.nn.functional as F
from typing import List, Tuple
from utils.model_loader import load_model
from model.fast_r2d2_functions import _convert_cache_to_tensor, force_encode, force_decode, multi_instance_learning


huge_neg = -1e7


class FastR2D2MIL(nn.Module):
    def __init__(self, config, label_num):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.parser = LSTMParser(config)
        enc_layer = nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, \
                                               dim_feedforward=config.intermediate_size, \
                                               batch_first=True)
        self.decoder = nn.TransformerEncoder(enc_layer, config.decoder_num_hidden_layers)
        self.role_embedding = nn.Parameter(torch.rand(3, config.hidden_size))
        self.attn_linear = nn.Linear(config.hidden_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, label_num),
        )
        self.norm = nn.InstanceNorm1d(config.hidden_size)

    def from_pretrain(self, r2d2_path, parser_path):
        self.r2d2.from_pretrain(r2d2_path)
        load_model(self.parser, parser_path)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None,
                labels: torch.Tensor = None,
                num_samples: int = -1):
        # input_ids: (N, max_id_len)
        if self.training:
            s_indices = self.parser(input_ids, attention_mask, atom_spans=atom_spans)
            results = self.r2d2(input_ids, attention_mask, merge_trajectories=s_indices,
                                sample_trees=num_samples, recover_tree=True, 
                                keep_tensor_cache=True,
                                lm_loss=True)

            bilm_loss = results['loss']

            if num_samples > 0:
                sampled_trees = results['sampled_trees']
                kl_loss = self.parser(input_ids, attention_mask,
                                    split_masks=sampled_trees['split_masks'],
                                    split_points=sampled_trees['split_points'])
            else:
                kl_loss = 0

        results = {}
        _, encoding_cache, root_nodes, _ = force_encode(self.parser, self.r2d2, input_ids, attention_mask, atom_spans)

        decoding_cache, root_nodes = force_decode(self.decoder, encoding_cache, root_nodes=root_nodes, \
                                                  role_embedding=self.role_embedding)

        weighted_logits, _, _ = multi_instance_learning(encoding_cache, decoding_cache, root_nodes, \
                                                      self.attn_linear, self.r2d2.device)
        # context : (N, max_id_len, dim)
        logits = self.mlp(weighted_logits)
        if self.training:
            loss = F.cross_entropy(logits, labels) + kl_loss + bilm_loss
            results['loss'] = loss
        results['roots'] = root_nodes
        results['logits'] = logits
        results['predict'] = logits
        return results


class FastR2D2MIML(nn.Module):
    def __init__(self, config, label_num):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.parser = LSTMParser(config)
        self.num_labels = label_num

        enc_layer = nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, \
                                               dim_feedforward=config.intermediate_size, \
                                               batch_first=True)
        self.decoder = nn.TransformerEncoder(enc_layer, config.decoder_num_hidden_layers)
        self.role_embedding = nn.Parameter(torch.rand(3, config.hidden_size))
        self.attn_linear = nn.Linear(config.hidden_size, label_num)

        self.multi_W1 = nn.Parameter(torch.rand(label_num, config.hidden_size, config.intermediate_size))
        self.active1 = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multi_W2 = nn.Parameter(torch.rand(label_num, config.intermediate_size, 1))
        self.multi_bias = nn.Parameter(torch.rand(label_num, 1))

        self.norm = nn.InstanceNorm1d(config.hidden_size)

    def from_pretrain(self, r2d2_path, parser_path):
        self.r2d2.from_pretrain(r2d2_path)
        load_model(self.parser, parser_path)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None,
                labels: List[List[int]] = None,
                num_samples: int = -1,
                bilm_loss: bool = False,
                **kwargs):
        # input_ids: (N, max_id_len)
        if self.training:#labels is not None:
            s_indices = self.parser(input_ids, attention_mask, atom_spans=atom_spans)
            results = self.r2d2(input_ids, attention_mask, merge_trajectories=s_indices,
                                sample_trees=num_samples, recover_tree=True, 
                                keep_tensor_cache=True,
                                lm_loss=True)

            bilm_loss = results['loss']

            if num_samples > 0:
                sampled_trees = results['sampled_trees']
                kl_loss = self.parser(input_ids, attention_mask,
                                    split_masks=sampled_trees['split_masks'],
                                    split_points=sampled_trees['split_points'])
            else:
                kl_loss = 0

        results = {}
        _, encoding_cache, root_nodes, _ = \
            force_encode(self.parser, self.r2d2, input_ids, attention_mask, atom_spans)

        # if self.apply_topdown:
        decoding_cache, root_nodes = force_decode(self.decoder, encoding_cache, root_nodes=root_nodes, \
                                                  role_embedding=self.role_embedding)

        attn_sqrt_dim = sqrt(decoding_cache.shape[-1])
        encoding_batch, decoding_batch, batch_masks, nodes_flatten_batch = \
            _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, self.r2d2.device)
        # batch_masks: [batch_size, max_len]

        attn_scores = self.attn_linear(decoding_batch) / attn_sqrt_dim  # [batch_size, max_len, label_num]
        attn_scores = attn_scores + (1.0 - batch_masks).unsqueeze(-1) * huge_neg
        attn_weights = F.softmax(attn_scores.permute(0, 2, 1), dim=-1)  # [batch_size, label_num, max_len]
        # apply mask
        weighted_logits = torch.einsum('bln,bni->bli', attn_weights, encoding_batch)  # [batch_size, label_num, dim]
        hidden = torch.einsum('bli,lij->blj', weighted_logits, self.multi_W1)  # [batch_size, label_num, dim]
        hidden = self.dropout(self.active1(hidden))
        logits = torch.einsum('bli,lij->blj', hidden, self.multi_W2)  # (batch_size, label_num, 1)
        logits = logits + self.multi_bias.unsqueeze(0)
        logits = logits.squeeze(2)

        if self.training: #labels is not None:
            target = np.full((input_ids.shape[0], self.num_labels), fill_value=0.0)
            for batch_i, intents_i in enumerate(labels):
                for intent_idx in intents_i:
                    target[batch_i][intent_idx] = 1
            target = torch.tensor(target, device=self.r2d2.device)
            loss = F.binary_cross_entropy_with_logits(logits, target) + kl_loss + bilm_loss
            results['loss'] = loss
        results['roots'] = root_nodes
        results['logits'] = logits
        results['predict'] = F.sigmoid(logits)
        results['attentions'] = attn_weights
        results['flatten_nodes'] = nodes_flatten_batch
        return results