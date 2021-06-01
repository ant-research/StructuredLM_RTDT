# coding=utf-8
# Copyright (c) 2021 Ant Group

import argparse
from typing import List

from transformers import AutoConfig, AutoTokenizer

from eval.tree_file_wrapper import WordPieceTreeFileWrapper, TreeFileWrapper
from utils.conll_utils import *
import torch
import numpy as np


class TreeEval:
    def __init__(self, conll_dir, max_len, len_division):
        self._conll_sentences = filter_conll_sentence(conll_dir, max_len, set([',']))
        self.lengh_division = len_division

    def eval(self, predict_model):
        sent_level_counter = {}
        corpus_level_counter = {}
        sent_count = 0
        for sent in self._conll_sentences:
            tokens = sent.tokens
            try:
                root, indices_mapping = predict_model(tokens)
            except Exception as e:
                print(e)
                continue

            if root is None:
                continue
            result = self.compare(root, sent, indices_mapping, ignore_punct=True)
            for k, (hit, total) in result.items():
                if total > 0:
                    acc = hit / total
                    sent_level_counter.setdefault(k, [])
                    sent_level_counter[k].append(acc)
                corpus_level_counter.setdefault(k, DotDict({'hit': 0, 'total': 0}))
                corpus_level_counter[k].hit += hit
                corpus_level_counter[k].total += total
            sent_count += 1
            if sent_count % 100 == 0:
                for k, vals in sent_level_counter.items():
                    print(f'sent level {k}: {np.array(vals).mean()}')
                for k, hit_total in corpus_level_counter.items():
                    print(f'corpus level {k}: {hit_total.hit / max(1, hit_total.total)}')

        print('done')

    def is_punct(self, conll_node):
        return conll_node.postag is None or conll_node.postag.lower() == 'punct' \
               or conll_node.postag.lower() == 'sym' \
               or conll_node.postag.lower() == 'x'

    def get_level(self, val, division):
        final_level = -1
        for level, depth_threshold in enumerate(division):
            if val <= depth_threshold:
                final_level = level
                break
        if final_level == -1:
            final_level = len(division)
        return final_level

    def count_span_valid(self, node, valid_start, valid_end, indices_mapping, conll_tree, ignore_punct):
        span_start, span_end = node.span
        if valid_start[span_start] and valid_end[span_end] and indices_mapping[span_start] != indices_mapping[span_end]:
            start_pos = indices_mapping[span_start]
            end_pos = indices_mapping[span_end]
            inner_nodes = set()
            in_degree_counter = {}
            out_degree_counter = {}
            out_degree_nodes = set()
            for i, node in enumerate(conll_tree.nodes[start_pos + 1: end_pos + 2]):
                assert node.node_id == start_pos + 1 + i
                if not (ignore_punct and self.is_punct(node)):
                    out_degree_counter[node.parent_id] = out_degree_counter.get(node.parent_id, 0) + 1
                    inner_nodes.add(node.node_id)
            for i, node in enumerate(conll_tree.nodes):
                if not (ignore_punct and self.is_punct(node)):
                    if node.parent_id in inner_nodes and node.node_id not in inner_nodes:
                        in_degree_counter[node.parent_id] = in_degree_counter.get(node.parent_id, 0) + 1
                    if node.node_id in inner_nodes and node.parent_id not in inner_nodes:
                        out_degree_nodes.add(node.node_id)
            for n_id in inner_nodes:
                if n_id in out_degree_counter:
                    out_degree_counter.pop(n_id)
            total_out = sum(out_degree_counter.values())
            include_all_children = True
            for _node_id, _in in in_degree_counter.items():
                if _node_id not in out_degree_nodes:
                    include_all_children = False
            if total_out <= 1 and include_all_children:
                return 1
        return 0

    def get_head(self, conll_tree, st, ed, ignore_punct):
        parent_nodes = set()
        inner_nodes = set()
        for node in conll_tree.nodes[st + 1: ed + 1]:
            if not (ignore_punct and self.is_punct(node)):
                inner_nodes.add(node.node_id)
                parent_nodes.add(node.parent_id)
        out_parents = parent_nodes - inner_nodes
        head_nodes = set()
        for node in conll_tree.nodes[st + 1: ed + 1]:
            if not (ignore_punct and self.is_punct(node)):
                if node.parent_id in out_parents:
                    head_nodes.add(node.node_id)
        if len(head_nodes) == 1:
            return list(head_nodes)[0]
        return -1

    def hit_relation(self, node, valid_start, valid_end, indices_mapping, conll_tree, ignore_punct):
        assert node.left is not None and node.right is not None
        # judge left and right is complete sub-tree
        left_span_start, left_span_end = node.left.span
        right_span_start, right_span_end = node.right.span
        if valid_start[left_span_start] and valid_end[left_span_end] \
                and valid_start[right_span_start] and valid_end[right_span_end]:
            left_token_start = indices_mapping[left_span_start]
            left_token_end = indices_mapping[left_span_end]

            right_token_start = indices_mapping[right_span_start]
            right_token_end = indices_mapping[right_span_end]

            head_left = self.get_head(conll_tree, left_token_start, left_token_end + 1, ignore_punct)
            head_right = self.get_head(conll_tree, right_token_start, right_token_end + 1, ignore_punct)
            if head_left != -1 and head_right != -1:
                if conll_tree.nodes[head_left].parent_id == head_right:
                    return head_left
                if conll_tree.nodes[head_right].parent_id == head_left:
                    return head_right
        return -1

    def check_word_chunk(self, node, valid_start, valid_end, indices_mapping, splited_token_ids):
        span_start, span_end = node.span
        if valid_start[span_start] and valid_end[span_end]:
            if indices_mapping[span_start] == indices_mapping[span_end] \
                    and indices_mapping[span_start] in splited_token_ids:
                return indices_mapping[span_start]
        return -1

    def check_propn_chunk(self, node, valid_start, valid_end, indices_mapping, nnp_span):
        span_start, span_end = node.span
        if valid_start[span_start] and valid_end[span_end]:
            token_st = indices_mapping[span_start]
            token_ed = indices_mapping[span_end]
            for _token_st, _token_ed in nnp_span:
                if _token_st == token_st + 1 and _token_ed == token_ed + 1:
                    return True
        return False

    def generate_boundary(self, indices_mapping, conll_tree, ignore_punct):
        valid_start = [False] * len(indices_mapping)
        valid_end = [False] * len(indices_mapping)
        assert max(indices_mapping) + 1 == len(conll_tree.nodes) - 1
        prev_token_id = -1
        for pos_i, token_id in enumerate(indices_mapping):
            if ignore_punct and self.is_punct(conll_tree.nodes[token_id + 1]):
                valid_start[pos_i] = True
                valid_end[pos_i] = True
            if prev_token_id != token_id:
                valid_start[pos_i] = True
                if pos_i - 1 >= 0:
                    valid_end[pos_i - 1] = True
            prev_token_id = token_id

        valid_end[-1] = True
        return valid_start, valid_end

    def get_splited_tokens(self, indices_mapping, conll_tree, ignore_punct):
        splited_token_ids = set()
        prev_token_id = -1
        for pos_i, token_id in enumerate(indices_mapping):
            if prev_token_id == token_id and not (ignore_punct and self.is_punct(conll_tree.nodes[token_id + 1])):
                splited_token_ids.add(token_id)
            prev_token_id = token_id

        return splited_token_ids

    def in_tree(self, conll, token_start, token_end):
        inner_nodes = set([node.node_id for node in conll.nodes[token_start:token_end+1]])
        out_degree = 0
        for node in conll.nodes[token_start: token_end + 1]:
            if node.parent_id not in inner_nodes:
                out_degree += 1
        return out_degree <= 1

    def get_NNP_spans(self, conll_tree):
        nnp_spans = []
        prev_is_propn = False
        current_span = DotDict({'start': -1, 'end': -1})
        for node in conll_tree.nodes[1:]:
            # find continous PROPN chunk and check if they are in the same tree
            if node.postag.lower() == 'propn' or node.postag.lower() == 'nnp':
                if prev_is_propn:
                    current_span.end = node.node_id
                else:
                    prev_is_propn = True
                    current_span.start = node.node_id
            else:
                if prev_is_propn:
                    prev_is_propn = False
                    if current_span.end != -1:
                        nnp_spans.append(current_span)
                    current_span = DotDict({'start': -1, 'end': -1})
        final_spans = []
        for span in nnp_spans:
            if self.in_tree(conll_tree, span.start, span.end):
                final_spans.append([span.start, span.end])
        return final_spans

    def compare(self, root, conll_tree, indices_mapping:List[int], ignore_punct=True):
        # iterate all brackets
        node_stack = [root]
        span_valid = 0

        token_hit = set()

        valid_start, valid_end = self.generate_boundary(indices_mapping, conll_tree, ignore_punct)
        splited_token_ids = self.get_splited_tokens(indices_mapping, conll_tree, ignore_punct)
        nnp_spans = self.get_NNP_spans(conll_tree)
        tokens = [n.token for n in conll_tree.nodes[1:]]
        len_level = self.get_level(len(tokens), self.lengh_division)

        propn_hit = 0

        while len(node_stack) > 0:
            current_node = node_stack.pop(0)
            is_leaf = True
            if current_node.left is not None:
                node_stack.append(current_node.left)
                is_leaf = False
            if current_node.right is not None:
                node_stack.append(current_node.right)
                is_leaf = False
            if is_leaf:
                continue

            # calculate start and end span
            start_pos, end_pos = current_node.span
            assert start_pos < end_pos

            hit_token_id = self.check_word_chunk(current_node, valid_start, valid_end,
                                                 indices_mapping, splited_token_ids)
            if hit_token_id != -1:
                token_hit.add(hit_token_id)

            hit_span = self.check_propn_chunk(current_node, valid_start, valid_end,
                                                 indices_mapping, nnp_spans)
            if hit_span:
                propn_hit += 1

            hit_inc = self.count_span_valid(current_node, valid_start, valid_end,
                                     indices_mapping, conll_tree, ignore_punct)
            span_valid += hit_inc

        assert len(token_hit) <= len(splited_token_ids)
        span_total = len(tokens) - 1
        assert span_valid <= span_total

        return {'global_span_valid': (span_valid, span_total),
                f'len_span_valid_{len_level}': (span_valid, span_total),
                'token_acc': (len(token_hit), len(splited_token_ids)),
                'NNP_acc': (propn_hit, len(nnp_spans))}


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Evaluating the compatibility of dependency tree')
    cmd.add_argument('--pred_tree_path', required=True, type=str)
    cmd.add_argument('--ground_truth_path', required=True, type=str)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--input_type', choices=['WD', 'WP'], help='WD for word level, WP for word piece level')
    options = cmd.parse_args()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(options.vocab_dir, use_fast=True)
    if options.input_type == 'WP':
        predictor = WordPieceTreeFileWrapper(options.pred_tree_path, tokenizer)
    else:
        predictor = TreeFileWrapper(options.pred_tree_path)
    evaluator = TreeEval(options.ground_truth_path, max_len=-1, len_division=[10, 20, 40])
    evaluator.eval(predictor)