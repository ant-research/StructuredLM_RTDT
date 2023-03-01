# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong

import torch
import  torch.nn.functional as F
from typing import List, Tuple
from model.fast_r2d2_functions import force_encode,force_decode
from model.fast_r2d2_dp_classification import FastR2D2DPClassification
from collections import deque


class FastR2D2DPClassificationShortcut(FastR2D2DPClassification):
    def __init__(self, config, total_lables_num, apply_topdown=False, exclusive=False, \
                 full_permutation=False, bilm=True):
        super().__init__(config, total_lables_num, apply_topdown=apply_topdown, exclusive=exclusive, \
                 full_permutation=full_permutation, bilm=bilm)
    
    def yield_labels_shortcut(self, root_nodes, logits, get_cache_id, traverse_all):
        max_label_id = torch.argmax(logits, dim=-1).cpu().data.numpy()
        softmax_logits = torch.softmax(logits, -1)
        labels_results = []
        nodes_interpretabel = []
        leaf_nodes_batch = []
        for root_node in root_nodes:
            labels = set()
            leaf_nodes = []
            node_queue = deque()
            node_interpretabel = root_node
            node_queue.append([root_node, False])
            setattr(root_node, "ancestor_scores", [softmax_logits.tolist()])
            while len(node_queue) > 0:
                current_node, met_terminal = node_queue.popleft()
                label_id = max_label_id[get_cache_id(current_node)]
                current_node.label = label_id
                setattr(current_node, "logits", softmax_logits[get_cache_id(current_node)].tolist())
                if label_id == self.nonterminal_label:
                    if not current_node.is_leaf:
                        node_queue.append([current_node.left, met_terminal])
                        node_queue.append([current_node.right, met_terminal])
                        setattr(current_node.left, "ancestor_scores", current_node.ancestor_scores + \
                                [softmax_logits[get_cache_id(current_node.left)].tolist()])
                        setattr(current_node.right, "ancestor_scores", current_node.ancestor_scores + \
                                [softmax_logits[get_cache_id(current_node.right)].tolist()])
                    else:
                        leaf_nodes.append(current_node)
                else:
                    if label_id != self.terminal_other_label:
                        if not met_terminal:
                            labels.add(label_id)
                        if current_node.j - current_node.i < node_interpretabel.j - node_interpretabel.i:
                            node_interpretabel = current_node
                    if traverse_all:
                        if current_node.left is not None \
                        and current_node.right is not None:
                            node_queue.append([current_node.left, True])
                            node_queue.append([current_node.right, True])
                            setattr(current_node.left, "ancestor_scores", current_node.ancestor_scores + \
                                [softmax_logits[get_cache_id(current_node.left)].tolist()])
                            setattr(current_node.right, "ancestor_scores", current_node.ancestor_scores + \
                                [softmax_logits[get_cache_id(current_node.right)].tolist()])
                        else:
                            leaf_nodes.append(current_node)
                            if current_node.label == 0:
                                pass
                                # print("there is a 0")

            nodes_interpretabel.append(node_interpretabel)
            labels_results.append(list(labels))
            leaf_nodes_batch.append(leaf_nodes)
        return labels_results, nodes_interpretabel, leaf_nodes_batch

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
            predicted_labels, nodes_interpretabel, leaf_nodes_batch= self.yield_labels_shortcut(root_nodes, cls_logits, get_cache_id, traverse_all)
            probs = torch.softmax(cls_logits, dim=-1)
            results = {}
            results['roots'] = root_nodes
            results['logits'] = cls_logits
            results['predict'] = predicted_labels
            results['node_interpretabel'] = nodes_interpretabel
            results['leaf_nodes_batch'] = leaf_nodes_batch
            return results