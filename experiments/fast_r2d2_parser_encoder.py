# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu


from collections import deque
import numpy as np
import torch.nn as nn
import torch
from model.r2d2_cuda import R2D2Cuda
import torch.nn.functional as F
from typing import List, Tuple
from utils.model_loader import load_model
from model.fast_r2d2_functions import force_encode_given_trees
from data_structure.r2d2_tree import PyNode


class TreeEncWithParser(nn.Module):
    def __init__(self, config, label_num):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.intermediate_size, label_num))

    def from_pretrain(self, model_path, parser_path=None):
        self.r2d2.from_pretrain(model_path)

    def load_model(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                atom_spans: List[List[Tuple[int]]] = None,
                root_nodes: List[PyNode] = None,
                labels: torch.Tensor = None,
                **kwargs):
        if self.training:
            # training
            e_ij, _, _ = force_encode_given_trees(root_nodes, self.r2d2, input_ids, attention_mask)
            logits = self.classifier(e_ij)
            force_encoding_loss = F.cross_entropy(logits, labels)# .mean()
            return {"loss": force_encoding_loss, "predict":logits}
        else:
            e_ij, _, _ = force_encode_given_trees(root_nodes, self.r2d2, input_ids, attention_mask)
            logits = self.classifier(e_ij)
            return {"predict": F.softmax(logits, dim=-1)}


class TreeEncMultiLabelWithParser(nn.Module):
    def __init__(self, config, label_num):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.num_labels = label_num
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.intermediate_size, label_num))

    def from_pretrain(self, model_path, parser_path=None):
        self.r2d2.from_pretrain(model_path)

    def load_model(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: List[List[int]] = None,
                root_nodes: List[PyNode] = None,
                **kwargs):
        if self.training:
            # training
            target = np.full((input_ids.shape[0], self.num_labels), fill_value=0.0)
            for batch_i, intents_i in enumerate(labels):
                for intent_idx in intents_i:
                    target[batch_i][intent_idx] = 1
            target = torch.tensor(target, device=input_ids.device)
            e_ij, _, _ = force_encode_given_trees(root_nodes, self.r2d2, input_ids, attention_mask)
            logits = self.classifier(e_ij)
            force_encoding_loss = F.binary_cross_entropy_with_logits(logits, target)# .mean()
            results = {}
            results['roots'] = None
            results['logits'] = logits
            results['loss'] = force_encoding_loss
            return results
        else:
            e_ij, _, _ = force_encode_given_trees(root_nodes, self.r2d2, input_ids, attention_mask)
            logits = self.classifier(e_ij)
            results = {}
            results['predict'] = torch.sigmoid(logits)
            return results


class ParserTreeEncDP(nn.Module):
    def __init__(self, config, label_num, exclusive=False):
        super().__init__()
        self.r2d2 = R2D2Cuda(config)
        self.num_labels = label_num
        self.exclusive = exclusive
        self.terminal_other_label = label_num
        self.nonterminal_label = label_num + 1
        self.total_labels_num = label_num + 2
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.intermediate_size, self.total_labels_num))

    def from_pretrain(self, model_path, parser_path=None):
        self.r2d2.from_pretrain(model_path)

    def load_model(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def dp(self, nodes_order, probs, target_labels, get_cache_id, atom_span, exclusive=False):
        root = nodes_order[-1]
        # p_yield = torch.zeros((root.j - root.i + 1, root.j - root.i + 1), 
        #                       dtype=torch.float, device=self.r2d2.device)
        p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        for node in nodes_order:
            if node.is_leaf or (node.i, node.j) in atom_span:
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

    def dp_others(self, nodes_order, probs, target_labels, get_cache_id):
        root = nodes_order[-1]
        p_yield = [[None for _ in range(root.j - root.i + 1)] for _ in range(root.j - root.i + 1)]
        target_labels = target_labels + [self.nonterminal_label, self.terminal_other_label]
        # target_labels, nonterminal_label, terminal_other_label are allowed
        for node in nodes_order:
            if node.is_leaf:
                # yield labels other than target labels
                p_yield[node.i][node.j] = 1 - probs[get_cache_id(node), target_labels].sum()
            else:
                left = node.left
                right = node.right
                p_yield[node.i][node.j] = 1 - probs[get_cache_id(node), target_labels].sum() + \
                                          probs[get_cache_id(node), self.nonterminal_label] * \
                                          (1 - (1 - p_yield[left.i][left.j]) * (1 - p_yield[right.i][right.j]))

        return p_yield[root.i][root.j]

    def yield_labels(self, root_nodes, logits, get_cache_id):
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
                    if current_node.left is not None \
                        and current_node.right is not None:
                        node_queue.append([current_node.left, True])
                        node_queue.append([current_node.right, True])
            labels_results.append(list(labels))
        return labels_results

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: List[List[int]] = None,
                atom_spans: List[List[Tuple[int]]] = None,
                root_nodes: List[PyNode] = None,
                **kwargs):
        if labels is not None:
            for batch_i, label_ids in enumerate(labels):
                labels[batch_i] = list(set(label_ids))
        batch_size = input_ids.shape[0]
        get_cache_id = lambda x: x.cache_id
        _, e_ij_cache, root_nodes = force_encode_given_trees(root_nodes, self.r2d2, input_ids, attention_mask)
        cls_logits = self.classifier(e_ij_cache)
        if self.training:
            # training
            log_p_sum = 0
            other_log_p = 0
            non_empty_count = 0
            loss = 0
            probs = F.softmax(cls_logits, dim=-1)
            if atom_spans is None:
                atom_spans = [[] for _ in range(batch_size)]
            for batch_i, root in enumerate(root_nodes):
                rev_order = deque()
                iter_queue = deque()
                iter_queue.append(root)
                while len(iter_queue) > 0:
                    current = iter_queue.popleft()
                    rev_order.appendleft(current)
                    if not current.is_leaf:
                        iter_queue.append(current.left)
                        iter_queue.append(current.right)
                
                # approximation
                if len(labels[batch_i]) > 0:
                    p_batch = self.dp(rev_order, probs, labels[batch_i], get_cache_id, atom_spans, self.exclusive)
                    log_p_sum += torch.log(torch.clamp(p_batch, min=1e-7)).sum(dim=-1)
                other_p = self.dp_others(rev_order, probs, labels[batch_i], get_cache_id)
                other_log_p += torch.log(torch.clamp(1 - other_p, min=1e-7))
                non_empty_count += 1

            if non_empty_count > 0:
                loss += (-log_p_sum - other_log_p) / non_empty_count
            results = {}
            results['roots'] = root_nodes
            results['logits'] = cls_logits
            results['loss'] = loss
            return results
        else:
            predicted_labels = self.yield_labels(root_nodes, cls_logits, get_cache_id)
            probs = torch.softmax(cls_logits, dim=-1)
            results = {}
            results['roots'] = root_nodes
            results['logits'] = cls_logits
            results['predict'] = predicted_labels
            return results