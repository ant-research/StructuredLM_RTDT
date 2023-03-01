# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

import numpy as np
import os
from collections import deque
from data_structure.const_tree import SpanTree



class BertForClassification(nn.Module):
    def __init__(self, pretrain_model_dir, num_labels) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrain_model_dir)
        self.encoder = AutoModel.from_pretrained(pretrain_model_dir)
        self.cls = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.intermediate_size),# nn.Linear(self.config.input_dim, self.config.intermediate_size),
                                 nn.GELU(),
                                 nn.Linear(self.config.intermediate_size, num_labels))


    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None,
                **kwargs):
        # ASSUME [CLS] is provided in input_ids
        results = self.encoder(input_ids, attention_mask)
        hidden_states = results.last_hidden_state
        logits = self.cls(hidden_states[:, 0, :])
        # logits = self.cls(results.pooler_output)
        
        if not self.training:
            return {'predict': torch.softmax(logits, dim=-1)}
        else:
            loss = F.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}

class BertForMultiIntent(nn.Module, ABC):
    def __init__(self, pretrain_model_dir, num_labels) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrain_model_dir)
        self.encoder = AutoModel.from_pretrained(pretrain_model_dir)
        
        self.cls = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.intermediate_size),# nn.Linear(self.config.input_dim, self.config.intermediate_size),
                                 nn.GELU(),
                                 nn.Linear(self.config.intermediate_size, num_labels))


    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: List[List[int]] = None,
                **kwargs):
        # ASSUME [CLS] is provided in input_ids
        results = self.encoder(input_ids, attention_mask)
        hidden_states = results.last_hidden_state
        logits = self.cls(hidden_states[:, 0, :])
        if not self.training:
            return {'predict': torch.sigmoid(logits)}
        else:
            target = np.full((input_ids.shape[0], self.num_labels), fill_value=0.0)
            for batch_i, intents_i in enumerate(labels):
                for intent_idx in intents_i:
                    target[batch_i][intent_idx] = 1
            target = torch.tensor(target, device=input_ids.device)
            loss = F.binary_cross_entropy_with_logits(logits, target)# .mean()
            return {'loss': loss, 'logits': logits}


class BertForDPClassification(nn.Module):
    def __init__(self, pretrain_model_dir, total_lables_num, exclusive=False):
        super().__init__()
        # pass in a trained supervised parser
        self.num_labels = total_lables_num
        self.config = AutoConfig.from_pretrained(pretrain_model_dir)
        self.encoder = AutoModel.from_pretrained(pretrain_model_dir)
        self.terminal_other_label = total_lables_num
        self.nonterminal_label = total_lables_num + 1
        self.total_labels_num = total_lables_num + 2
        self.exclusive = exclusive
        self.cls = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.intermediate_size),
                                 nn.GELU(),
                                 nn.Dropout(self.config.hidden_dropout_prob),
                                 nn.Linear(self.config.intermediate_size, self.total_labels_num))

    def dp_target(self, probs, reverse_node_order, tgt):
        root = reverse_node_order[-1]
        p_yield = [[None for _ in range(root.ed - root.st + 1)] for _ in range(root.ed - root.st + 1)]
        for node in reverse_node_order:
            if len(node.subtrees) == 0:
                p_yield[node.st][node.ed] = probs[node.cache_id, tgt]
            else:
                assert p_yield[node.st][node.ed] is None
                if not self.exclusive:
                    no_target_prob = 1
                    for sub_tree in node.subtrees:
                        no_target_prob = no_target_prob * (1 - p_yield[sub_tree.st][sub_tree.ed])
                    p_yield[node.st][node.ed] = probs[node.cache_id, tgt] + \
                                            probs[node.cache_id, self.nonterminal_label] * \
                                            (1 - no_target_prob)
                else:
                    exclusive_p_sum = 0
                    for child_idx in range(len(node.subtrees)):
                        exclusive_p = 1
                        for sub_tree_idx, sub_tree in enumerate(node.subtrees):
                            if child_idx == sub_tree_idx:
                                exclusive_p = exclusive_p * p_yield[sub_tree.st][sub_tree.ed]
                            else:
                                exclusive_p = exclusive_p * (1 - p_yield[sub_tree.st][sub_tree.ed])
                        exclusive_p_sum += exclusive_p

                    p_yield[node.st][node.ed] = probs[node.cache_id, tgt] + \
                                            probs[node.cache_id, self.nonterminal_label] * \
                                            exclusive_p_sum

        return p_yield[root.st][root.ed]

    
    def dp_others(self, probs, reverse_node_order, tgts):
        root = reverse_node_order[-1]
        p_yield = [[None for _ in range(root.ed - root.st + 1)] for _ in range(root.ed - root.st + 1)]
        other_labels = tgts + [self.nonterminal_label, self.terminal_other_label]
        for node in reverse_node_order:
            if len(node.subtrees) == 0:
                p_yield[node.st][node.ed] = 1 - probs[node.cache_id, other_labels].sum()
            else:
                assert p_yield[node.st][node.ed] is None
                if not self.exclusive:
                    all_other_prob = 1
                    for sub_tree in node.subtrees:
                        # all sub_tree generate target labels
                        all_other_prob = all_other_prob * p_yield[sub_tree.st][sub_tree.ed]
                    p_yield[node.st][node.ed] = 1 - probs[node.cache_id, other_labels].sum() + \
                                            probs[node.cache_id, self.nonterminal_label] * \
                                            (1 - all_other_prob)
                else:
                    exclusive_p_sum = 0
                    for child_idx in range(len(node.subtrees)):
                        exclusive_p = 1
                        for sub_tree_idx, sub_tree in enumerate(node.subtrees):
                            if child_idx == sub_tree_idx:
                                exclusive_p = exclusive_p * p_yield[sub_tree.st][sub_tree.ed]
                            else:
                                exclusive_p = exclusive_p * (1 - p_yield[sub_tree.st][sub_tree.ed])
                        exclusive_p_sum += exclusive_p

                    p_yield[node.st][node.ed] = 1 - probs[node.cache_id, other_labels].sum() + \
                                            probs[node.cache_id, self.nonterminal_label] * \
                                            exclusive_p_sum

        return p_yield[root.st][root.ed]

    def yield_labels(self, root_nodes, logits):
        max_label_id = torch.argmax(logits, dim=-1).cpu().data.numpy()
        labels_results = []
        for root_node in root_nodes:
            labels = set()
            node_queue = deque()
            node_queue.append(root_node)
            while len(node_queue) > 0:
                current_node = node_queue.popleft()
                label_id = max_label_id[current_node.cache_id]
                current_node.label = label_id
                if label_id == self.nonterminal_label:
                    for subtree in current_node.subtrees:
                        node_queue.append(subtree)
                elif label_id != self.terminal_other_label:
                    labels.add(label_id)
            labels_results.append(list(labels))
        return labels_results
        

    @abstractmethod
    def forward(self, input_ids, attention_mask, trees: List[SpanTree], 
                labels: List[List[int]] = None,
                **kwargs):
        pass


class BertForDPClassificationMeanPooling(BertForDPClassification):
    def __init__(self, pretrain_model_dir, total_lables_num, exclusive=False):
        super().__init__(pretrain_model_dir, total_lables_num, exclusive)

    def forward(self, input_ids, attention_mask, trees: List[SpanTree], 
                labels: List[List[int]] = None,
                **kwargs):
        # roots of constituency trees
        results = self.encoder(input_ids, attention_mask)
        hidden_states = results.last_hidden_state
        dim = hidden_states.shape[-1]
        zero_padding = torch.zeros((input_ids.shape[0], 1, dim), device=input_ids.device)
        padded_hidden = torch.cat([hidden_states, zero_padding], dim=1)
        max_len = hidden_states.shape[1]  # also padding pos

        gather_indices_batch = []
        max_span_num = 0
        span_len_batch = []
        for sent_i, span_tree in enumerate(trees):
            node_queue = deque()
            node_queue.append(span_tree)
            span_gather_indices = []
            span_lens = []
            while len(node_queue) > 0:
                current_node = node_queue.popleft()
                gather_indices = []
                assert current_node.ed >= current_node.st
                for pos_i in range(current_node.st, current_node.ed + 1):
                    gather_indices.append(pos_i)
                span_lens.append(current_node.ed - current_node.st + 1)
                assert len(gather_indices) <= max_len
                gather_indices = gather_indices + \
                    [max_len] * (max_len - len(gather_indices))
                span_gather_indices.append(gather_indices)
                for subtree in current_node.subtrees:
                    node_queue.append(subtree)
            max_span_num = max(len(span_gather_indices), max_span_num)
            gather_indices_batch.append(span_gather_indices)
            span_len_batch.append(span_lens)

        # padding gather_indices_batch
        for span_gather_indices in gather_indices_batch:
            padding_indices = [[max_len] * max_len] * (max_span_num - len(span_gather_indices))
            span_gather_indices.extend(padding_indices)
        
        for span_lens in span_len_batch:
            span_lens.extend([1] * (max_span_num - len(span_lens)))
        
        gather_indices_batch = torch.tensor(gather_indices_batch, device=input_ids.device)
        # (batch_i, max_span_num, max_len)
        gather_indices_batch = gather_indices_batch.unsqueeze(-1).repeat(1, 1, 1, dim)
        gathered_hidden = padded_hidden.unsqueeze(1).\
            repeat(1, max_span_num, 1, 1).gather(2, gather_indices_batch)
        hidden_sum = gathered_hidden.sum(dim=2)  # (batch_i, max_span_num, dim)
        span_len_batch = torch.tensor(span_len_batch, dtype=torch.float, device=input_ids.device)
        hidden_mean = hidden_sum / span_len_batch.unsqueeze(2)  # (batch_i, max_span_num, dim)
        tensor_cache = hidden_mean.view(-1, dim)  # (batch_size * max_span_num, dim)
        logits = self.cls(tensor_cache)
        probs = F.softmax(logits, dim=-1)  # (batch_size * max_span_num, label_num)

        if self.training: #labels is not None:
            log_p_sum = 0
            for sent_i, span_tree in enumerate(trees):
                node_queue = deque()
                reverse_order = deque()
                node_queue.append(span_tree)
                cache_id_offset = sent_i * max_span_num
                while len(node_queue) > 0:
                    current_node = node_queue.popleft()
                    reverse_order.appendleft(current_node)
                    current_node.cache_id = cache_id_offset
                    cache_id_offset += 1
                    for subtree in current_node.subtrees:
                        node_queue.append(subtree)

                target_lop_p_sum = 0
                for tgt in labels[sent_i]:
                    target_lop_p_sum += self.dp_target(probs, reverse_order, tgt)
                other_p = self.dp_others(probs, reverse_order, labels[sent_i])
                if len(labels[sent_i]) != 0:
                    log_p_sum += torch.log(torch.clamp(target_lop_p_sum / len(labels[sent_i]), min=1e-7))
                log_p_sum += torch.log(torch.clamp(1 - other_p, min=1e-7))
            loss = -log_p_sum / len(trees)
            results = {}
            results['roots'] = trees
            results['loss'] = loss
            results['logits'] = logits
        else:
            for sent_i, span_tree in enumerate(trees):
                node_queue = deque()
                reverse_order = deque()
                node_queue.append(span_tree)
                cache_id_offset = sent_i * max_span_num
                while len(node_queue) > 0:
                    current_node = node_queue.popleft()
                    reverse_order.appendleft(current_node)
                    current_node.cache_id = cache_id_offset
                    cache_id_offset += 1
                    for subtree in current_node.subtrees:
                        node_queue.append(subtree)
            predict_labels = self.yield_labels(trees, logits)
            results = {}
            results['roots'] = trees
            results['logits'] = logits
            results['predict'] = predict_labels
        return results