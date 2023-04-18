# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import numpy as np
import torch.nn as nn
import torch
import  torch.nn.functional as F
from typing import List, Tuple
from model.r2d2_common import CacheSlots
from model.r2d2_cuda import R2D2Cuda
from model.topdown_parser import LSTMParser
from utils.model_loader import load_model
from model.fast_r2d2_functions import force_encode,force_decode
from collections import deque


def _is_noise(labels):
    return labels == [-1]


def convert_value_to_binary(val, total_label, indices=None):
    if indices is None:
        indices = [_ for _ in range(total_label)]
    binary_indices = [0] * len(indices)

    for bit_pos in range(len(indices)):
        if val % 2 == 1:
            binary_indices[-1 - bit_pos] = 1
        else:
            binary_indices[-1 - bit_pos] = 0
        val = val // 2

    return binary_indices

def convert_binary_to_value(binary, total_label, indices=None):
    if indices is None:
        indices = [_ for _ in range(total_label)]
    assert len(binary) == len(indices)
    result = 0
    full_binary = [0] * total_label
    for bin_idx, bin_val in zip(indices, binary):
        full_binary[bin_idx] = bin_val
    for val in full_binary:
        result *= 2
        result += val
    
    return result

def convert_to_std_binary(local_binary, standard_binary):
    convert_result = []
    offset = 0
    for val in standard_binary:
        if val == 0:
            convert_result.append(0)
        elif val == 1:
            convert_result.append(local_binary[offset])
            offset += 1
    assert len(convert_result) == len(standard_binary)
    return convert_result

def satisfy(left_binary, right_binary):
    for left_val, right_val in zip(left_binary, right_binary):
        if left_val == 0 and right_val == 0:
            return False
    return True


def flip(binary):
    flip_binary = []
    for val in binary:
        flip_binary.append(1 - val)
    return flip_binary


class FastR2D2DPClassification(nn.Module):
    def __init__(self, config, total_lables_num, apply_topdown=False, exclusive=False, \
                 full_permutation=False, bilm=True):
        '''
        total_labels_num: the total num of task defined labels
        apply_topdown: whether a top-down encoder is applied
        exclusive: whether a label is associated with a single span.
        full_permutation: whether using dynamic programming considering all potential combinations
        bilm: whether training downstream tasks with bilm loss
        '''
        super().__init__()
        # there are two meta label: nonterminal_label, terminal_other_label
        # nonterminal: go on visiting children nodes
        # terminal_other_label: stop and add nothong
        self.terminal_other_label = total_lables_num
        self.nonterminal_label = total_lables_num + 1
        self.total_labels_num = total_lables_num + 2
        self.r2d2 = R2D2Cuda(config)
        self.parser = LSTMParser(config)
        self.cls = nn.Sequential(nn.Linear(config.embedding_dim, config.hidden_size),
                                 nn.GELU(),
                                 nn.Dropout(config.hidden_dropout_prob),
                                 nn.Linear(config.hidden_size, self.total_labels_num))
        self.apply_topdown = apply_topdown
        self.exclusive = exclusive
        self.full_permutation = full_permutation
        self.bilm = bilm
        if self.apply_topdown:
            self.role_embeddings = nn.Parameter(torch.rand([3, config.embedding_dim]))
            self.root_embedding = nn.parameter.Parameter(torch.zeros([config.embedding_dim]))
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.num_attention_heads,
                                                       dim_feedforward=config.hidden_size,
                                                       dropout=config.hidden_dropout_prob,
                                                       activation='gelu',
                                                       batch_first=True)
            self.decoder = nn.TransformerEncoder(encoder_layer, config.decoder_num_hidden_layers)
    
    def from_pretrain(self, model_path, parser_path):
        self.r2d2.from_pretrain(model_path)
        load_model(self.parser, parser_path)

    def load_model(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()


    def dp_valid_only(self, nodes_order, probs, target_labels, get_cache_id, current_atom_spans, exclusive=True):
        root = nodes_order[-1]
        # p_yield = torch.zeros((root.j - root.i + 1, root.j - root.i + 1), 
        #                       dtype=torch.float, device=self.r2d2.device)
        valid_p_sum = probs[:, target_labels + [self.terminal_other_label]].sum(dim=-1, keepdim=True)
        _p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        # the yield result is subset of target_labels but without the specified one.
        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in current_atom_spans:
                _p_yield[node.i][node.j] = valid_p_sum[get_cache_id(node)] - probs[get_cache_id(node), target_labels]
                
            else:
                left = node.left
                right = node.right
                left_other = _p_yield[left.i][left.j]
                right_other = _p_yield[right.i][right.j]
                terminal_valid = valid_p_sum[get_cache_id(node)] - probs[get_cache_id(node), target_labels]
                assert _p_yield[node.i][node.j] is None
                assert terminal_valid.shape[0] == len(target_labels), f'{terminal_valid.shape[0]} / {len(target_labels)}'
                _p_yield[node.i][node.j] = terminal_valid + probs[get_cache_id(node), self.nonterminal_label] * \
                                           (left_other * right_other)
            assert _p_yield[node.i][node.j].shape[0] == len(target_labels), f'{_p_yield[node.i][node.j].shape[0]} / {len(target_labels)}'
            
        

        p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in current_atom_spans:
                p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels]
            else:
                left = node.left
                right = node.right
                left_hit = p_yield[left.i][left.j]
                left_others = _p_yield[left.i][left.j]
                right_hit = p_yield[right.i][right.j]
                right_others = _p_yield[right.i][right.j]
                assert p_yield[node.i][node.j] is None
                if not exclusive:
                    p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels] + \
                                            probs[get_cache_id(node), self.nonterminal_label] * \
                                            (left_hit * right_others + left_others * right_hit + left_hit * right_hit)
                else:
                    p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels] + \
                                            probs[get_cache_id(node), self.nonterminal_label] * \
                                            (left_hit * right_others + left_others * right_hit)

        return p_yield[root.i][root.j]

    def dp_exponential(self, nodes_order, probs, target_labels, get_cache_id, current_atom_spans):
        total_state_space = pow(2, len(target_labels))
        root = nodes_order[-1]
        total_label = len(target_labels)

        p_yield = [[[0] * total_state_space for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        # yield 包含指定标签的所有子树概率之和

        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in current_atom_spans:
                for binary_idx in range(total_label):
                    value = convert_binary_to_value([1], total_label, [binary_idx])
                    p_yield[node.i][node.j][value] = probs[get_cache_id(node), target_labels[binary_idx]]
                
                p_yield[node.i][node.j][0] = probs[get_cache_id(node), [self.nonterminal_label, self.terminal_other_label]].sum()
                # p_yield[node.i][node.j][0] = 1 # no constraints
            else:
                left = node.left
                right = node.right
                p_yield[node.i][node.j][0] = probs[get_cache_id(node), self.terminal_other_label] + probs[get_cache_id(node), self.nonterminal_label] * \
                                                (p_yield[left.i][left.j][0] * p_yield[right.i][right.j][0])
                for state_idx in range(1, total_state_space):
                    state_binary = convert_value_to_binary(state_idx, total_label)
                    sub_state_indices = []
                    for ind_idx, state in enumerate(state_binary):
                        if state == 1:
                            sub_state_indices.append(ind_idx)
                    total_sub_space = convert_binary_to_value([1] * len(sub_state_indices), \
                                                                total_label, sub_state_indices)
                    if len(sub_state_indices) == 1:
                        left_hit = p_yield[left.i][left.j][state_idx]
                        right_hit = p_yield[right.i][right.j][state_idx]
                        left_other = p_yield[left.i][left.j][0]
                        right_other = p_yield[right.i][right.j][0]
                        target_label = target_labels[sub_state_indices[0]]
                        nonterminal_case =  left_hit * right_other + left_other * right_hit + left_hit * right_hit
                        # nonterminal_case = 1 - (1 - left_hit) * (1 - right_hit)
                        p_yield[node.i][node.j][state_idx] = probs[get_cache_id(node), target_label] + \
                            probs[get_cache_id(node), self.nonterminal_label] * nonterminal_case
                    else:
                        for left_part_state_id in range(0, total_sub_space + 1):
                            for right_part_state_id in range(0, total_sub_space + 1):
                                left_binary = convert_value_to_binary(left_part_state_id, total_label, \
                                                                        indices=sub_state_indices)
                                right_binary = convert_value_to_binary(right_part_state_id, total_label, \
                                                                        indices=sub_state_indices)
                                if satisfy(left_binary, right_binary):
                                    # aligh with standard binary
                                    std_left = convert_to_std_binary(left_binary, state_binary)
                                    std_right = convert_to_std_binary(right_binary, state_binary)
                                    left_id = convert_binary_to_value(std_left, total_label)
                                    right_id = convert_binary_to_value(std_right, total_label)
                                    p_yield[node.i][node.j][state_idx] += probs[get_cache_id(node), self.nonterminal_label] * \
                                        p_yield[left.i][left.j][left_id] * p_yield[right.i][right.j][right_id]

        if node.i < node.j:
            return p_yield[node.i][node.j][-1]
        else:
            return torch.zeros([1], device=self.r2d2.device)


    def dp(self, nodes_order, probs, target_labels, get_cache_id, current_atom_spans, exclusive=False):
        root = nodes_order[-1]
        p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in current_atom_spans:
                p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels]
            else:
                left = node.left
                right = node.right
                x = p_yield[left.i][left.j]
                y = p_yield[right.i][right.j]
                assert p_yield[node.i][node.j] is None
                assert x is not None
                assert y is not None
                if not exclusive:
                    p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels] + \
                                            probs[get_cache_id(node), self.nonterminal_label] * \
                                            (x + y - x*y)
                else:
                    p_yield[node.i][node.j] = probs[get_cache_id(node), target_labels] + \
                                            probs[get_cache_id(node), self.nonterminal_label] * \
                                            (x + y - 2*x*y)

        return p_yield[root.i][root.j]

    def dp_others(self, nodes_order, probs, target_labels, get_cache_id, current_atom_spans):
        root = nodes_order[-1]
        p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        target_labels = target_labels + [self.nonterminal_label, self.terminal_other_label]
        # target_labels, nonterminal_label, terminal_other_label are allowed
        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in current_atom_spans:
                # yield labels other than target labels
                p_yield[node.i][node.j] = 1 - probs[get_cache_id(node), target_labels].sum()
            else:
                left = node.left
                right = node.right
                p_yield[node.i][node.j] = 1 - probs[get_cache_id(node), target_labels].sum() + \
                                          probs[get_cache_id(node), self.nonterminal_label] * \
                                          (1 - (1 - p_yield[left.i][left.j]) * (1 - p_yield[right.i][right.j]))

        return p_yield[root.i][root.j]

    def nt_sum_p(self, nodes_order, probs, get_cache_id):
        cache_ids = []
        for node in nodes_order:
            cache_ids.append(get_cache_id(node))
        p_square = torch.square(1 - probs[cache_ids, self.nonterminal_label])

        return p_square.mean()

    def yield_labels(self, root_nodes, logits, get_cache_id, traverse_all):
        max_label_id = torch.argmax(logits, dim=-1).cpu().data.numpy()
        labels_results = []
        for root_node in root_nodes:
            labels = set()
            node_queue = deque()
            node_queue.append([root_node, False])
            while len(node_queue) > 0:
                current_node, met_terminal = node_queue.popleft()
                label_id = max_label_id[get_cache_id(current_node)]
                current_node.label = label_id
                if label_id == self.nonterminal_label:
                    if not current_node.is_leaf:
                        node_queue.append([current_node.left, met_terminal])
                        node_queue.append([current_node.right, met_terminal])
                else:
                    if label_id != self.terminal_other_label and not met_terminal:
                        labels.add(label_id)
                    if traverse_all and current_node.left is not None \
                        and current_node.right is not None:
                        node_queue.append([current_node.left, True])
                        node_queue.append([current_node.right, True])
            labels_results.append(list(labels))
        return labels_results

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                num_samples: int = 256,
                atom_spans: List[List[Tuple[int]]] = None,
                labels: List[List[int]] = None,
                traverse_all=False,
                **kwargs):
        '''
        intents: 
        '''
        # remove duplicated labels
        if labels is not None:
            for batch_i, label_ids in enumerate(labels):
                labels[batch_i] = list(set(label_ids))
        batch_size = input_ids.shape[0]
        # max_seq_len = input_ids.shape[1]
        _, e_ij_cache, root_nodes, _ = force_encode(self.parser, self.r2d2, \
            input_ids, attention_mask=attention_mask, atom_spans=atom_spans)
        if not self.apply_topdown:
            cls_logits = self.cls(e_ij_cache)  # classification logits
            get_cache_id = lambda x: x.cache_id
        else:
            decoding_cache, _ = force_decode(self.decoder, e_ij_cache, root_nodes, 
                                             root_role_embedding=self.root_embedding,
                                             role_embedding=self.role_embeddings)
            cls_logits = self.cls(decoding_cache)
            get_cache_id = lambda x: x.decode_cache_id

        if self.training:
            # add bilm loss
            loss = 0
            if self.bilm:
                with torch.no_grad():
                    s_indices = self.parser(input_ids, attention_mask, atom_spans=atom_spans)
                results = self.r2d2(input_ids, attention_mask, merge_trajectories=s_indices,
                                    sample_trees=num_samples)
                sampled_trees = results['sampled_trees']
                bilm_loss = results['loss']
                kl_loss = self.parser(input_ids, attention_mask,
                                    split_masks=sampled_trees['split_masks'],
                                    split_points=sampled_trees['split_points'])
                loss = bilm_loss + kl_loss
            probs = F.softmax(cls_logits, dim=-1)
            log_p_items = []
            noise_items = []

            if atom_spans is None:
                atom_spans = [[] for _ in range(batch_size)]
            for batch_i, root in enumerate(root_nodes):
                rev_order = deque()
                iter_queue = deque()
                current_atom_spans = set(atom_spans[batch_i])
                iter_queue.append(root)
                while len(iter_queue) > 0:
                    current = iter_queue.popleft()
                    rev_order.appendleft(current)
                    if not current.is_leaf and \
                        (current.i, current.j) not in current_atom_spans:
                        iter_queue.append(current.left)
                        iter_queue.append(current.right)
                if not _is_noise(labels[batch_i]):
                    if self.full_permutation:
                        # full permutation mode
                        if len(labels[batch_i]) > 0:
                            p_batch = self.dp_exponential(rev_order, probs, labels[batch_i], get_cache_id, \
                                                            current_atom_spans)
                            log_p_items.append(torch.log(torch.clamp(p_batch, min=1e-7)).sum(dim=-1))
                    else:
                        # approximation
                        log_p = 0
                        if len(labels[batch_i]) > 0:
                            p_batch = self.dp(rev_order, probs, labels[batch_i], get_cache_id, current_atom_spans, self.exclusive)
                            log_p += torch.log(torch.clamp(p_batch, min=1e-7)).sum(dim=-1)
                        other_p = self.dp_others(rev_order, probs, labels[batch_i], get_cache_id, current_atom_spans)
                        log_p += torch.log(torch.clamp(1 - other_p, min=1e-7))
                        log_p_items.append(-log_p)
                else:
                    # Only used to train SST-2 with noise
                    noise_items.append(self.nt_sum_p(rev_order, probs, get_cache_id))
                    
            if len(log_p_items) > 0:
                loss += torch.stack(log_p_items).mean()
            if len(noise_items) > 0:
                loss += torch.stack(noise_items).mean()
            results = {}
            results['roots'] = root_nodes
            results['logits'] = cls_logits
            results['loss'] = loss
            return results
        else:
            # return cls_logits, root_nodes
            predicted_labels = self.yield_labels(root_nodes, cls_logits, get_cache_id, traverse_all)
            probs = torch.softmax(cls_logits, dim=-1)
            results = {}
            results['roots'] = root_nodes
            results['logits'] = cls_logits
            results['predict'] = predicted_labels
            return results


class FastR2D2MultiLabelRoot(nn.Module):
    def __init__(self, config, label_num, transformer_parser=False):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.num_labels = label_num
        self.parser = LSTMParser(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.intermediate_size, label_num))

    def from_pretrain(self, model_path, parser_path):
        self.r2d2.from_pretrain(model_path)
        load_model(self.parser, parser_path)

    def load_model(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                num_samples: int = 0,
                atom_spans: List[List[Tuple[int]]] = None,
                labels: List[List[int]] = None,
                force_encoding=False,
                **kwargs):
        if self.training:
            # training
            target = np.full((input_ids.shape[0], self.num_labels), fill_value=0.0)
            for batch_i, intents_i in enumerate(labels):
                for intent_idx in intents_i:
                    target[batch_i][intent_idx] = 1
            target = torch.tensor(target, device=input_ids.device)
            s_indices = self.parser(input_ids, attention_mask, atom_spans=atom_spans)
            results = self.r2d2(input_ids, attention_mask, merge_trajectories=s_indices,
                                sample_trees=num_samples, recover_tree=True, keep_tensor_cache=True)
            tables = results['tables']
            tensor_cache = results['tensor_cache']
            sampled_trees = results['sampled_trees']

            root_cache_ids = []
            for t in tables:
                root_cache_ids.append(t.root.best_node.cache_id)
            e_ij = tensor_cache.gather(root_cache_ids, [CacheSlots.E_IJ])[0]
            logits = self.classifier(e_ij)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            bilm_loss = results['loss']
            kl_loss = self.parser(input_ids, attention_mask,
                                split_masks=sampled_trees['split_masks'],
                                split_points=sampled_trees['split_points'])
            
            # force encoding
            e_ij, _, _, _ = force_encode(self.parser, self.r2d2, input_ids, attention_mask, atom_spans)
            logits = self.classifier(e_ij)
            force_encoding_loss = F.binary_cross_entropy_with_logits(logits, target)# .mean()
            total_loss = force_encoding_loss + loss + kl_loss + bilm_loss
            results = {}
            results['roots'] = None
            results['logits'] = logits
            results['loss'] = total_loss
            return results
        else:
            # Implement two mode for inference
            if not force_encoding:
                s_indices = self.parser(input_ids, attention_mask, atom_spans=atom_spans)
                results = self.r2d2(input_ids, attention_mask, merge_trajectories=s_indices,
                                    recover_tree=True, keep_tensor_cache=True)
                tables = results['tables']
                tensor_cache = results['tensor_cache']
                root_cache_ids = []
                for t in tables:
                    root_cache_ids.append(t.root.best_node.cache_id)
                e_ij = tensor_cache.gather(root_cache_ids, [CacheSlots.E_IJ])[0]
            else:
                e_ij, _, _, _ = force_encode(self.parser, self.r2d2, input_ids, attention_mask, atom_spans)
            logits = self.classifier(e_ij)
            results = {}
            results['predict'] = torch.sigmoid(logits)
            return results