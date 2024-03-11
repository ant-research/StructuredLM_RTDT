# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang

from data_structure.const_tree import ConstTree
from data_structure.basic_structure import DotDict
import logging
import codecs
import argparse
from eval.tree_file_wrapper import TreeFileWrapper


def get_span(st, ed, indices_mapping):
    left = list(map(lambda x: x[0], filter(lambda x: x[1] == st, enumerate(indices_mapping))))
    left = left[0] if len(left) == 1 else min(*left)
    right = list(map(lambda x: x[0], filter(lambda x: x[1] == ed, enumerate(indices_mapping))))
    right = right[0] if len(right) == 1 else max(*right)
    assert isinstance(left, int) and isinstance(right, int)
    return left, right


class ConstituentsEval:
    def __init__(self, data_path, constituents):
        self._trees = self.parse_trees(data_path)
        self._constituents = constituents

    @staticmethod
    def parse_trees(path):
        trees = []
        with codecs.open(path, mode='r', encoding='utf-8') as f:
            for l in f:
                if len(l) > 0:
                    try:
                        result = ConstTree.from_string(l)
                        if len(result[1]) > 1:
                            trees.append(result)
                    except Exception as exp:
                        logging.exception(exp)
        return trees

    def collect_NNP_spans(self, tree, indices_mapping):
        start = -1
        end = -1
        spans = set()
        for child in tree.children:
            if isinstance(child, ConstTree):
                if child.span[1] - child.span[0] == 1 and child.tag.lower() == 'nnp':
                    if start == -1:
                        start = child.span[0]
                    end = child.span[1] - 1
                else:
                    if start != -1 and end != -1 and end > start:
                        l, r = get_span(start, end, indices_mapping)
                        spans.add((l, r))
                    start = -1
                    end = -1
                spans |= self.collect_NNP_spans(child, indices_mapping)
        if start != -1 and end != -1 and end > start:
            l, r = get_span(start, end, indices_mapping)
            spans.add((l, r))
        return spans

    def collect_WP_spans(self, indices_mapping):
        prev_token_idx = -1
        span_len = 0
        start = -1
        span_set = set()
        for idx, token_idx in enumerate(indices_mapping):
            if token_idx == prev_token_idx:
                span_len += 1
            else:
                if span_len > 0:
                    assert start != idx
                    span_set.add((start, start + span_len))
                span_len = 0
                start = idx
            prev_token_idx = token_idx
        if span_len > 0:
            span_set.add((start, start + span_len))
        return span_set

    def _is_span(self, node):
        current_node = node
        while isinstance(current_node, ConstTree):
            if len(current_node.children) > 1:
                return True
            current_node = current_node.children[0]
        return False

    def collect_constituents_spans(self, tree, indices_mapping, span_name):
        queue = [tree]
        span_set = set()
        while len(queue) > 0:
            current_node = queue.pop(0)
            if isinstance(current_node, ConstTree):
                if current_node.tag.lower() == span_name.lower():
                    st, ed = current_node.span
                    if ed - st <= 1:
                        continue
                    ed -= 1
                    left, right = get_span(st, ed, indices_mapping)
                    span_set.add((left, right))
                for child in current_node.children:
                    queue.append(child)
        return span_set

    def eval(self, predict_model):
        hit_total = {}
        hit_total['WP'] = DotDict({'hit': 0, 'total': 0})
        hit_total['NNP'] = DotDict({'hit': 0, 'total': 0})
        for c in self._constituents:
            hit_total[c] = DotDict({'hit': 0, 'total': 0})
        for tree, lexicons in self._trees:
            tokens = [l.string for l in lexicons]
            root, indices_mapping = predict_model(tokens)
            root = root[0]

            if root is None:
                continue

            wp_spans = self.collect_WP_spans(indices_mapping)
            nnp_spans = self.collect_NNP_spans(tree, indices_mapping)
            constituents_spans = DotDict()
            for c in self._constituents:
                constituents_spans[c] = self.collect_constituents_spans(tree, indices_mapping, c)

            binary_tree_spans = root.to_spans()
            hit_total['WP'].hit += len(wp_spans & binary_tree_spans)
            hit_total['WP'].total += len(wp_spans)
            hit_total['NNP'].hit += len(nnp_spans & binary_tree_spans)
            hit_total['NNP'].total += len(nnp_spans)
            for c in self._constituents:
                hit_total[c].hit += len(constituents_spans[c] & binary_tree_spans)
                hit_total[c].total += len(constituents_spans[c])
        for name, kv in hit_total.items():
            if kv.total == 0:
                print(f'{name}: 0')
            else:
                print(f'{name}: {kv.hit/kv.total}')
                
                
if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Parameter of eval constituents')
    cmd.add_argument('--tree_path', default=None, type=str)
    cmd.add_argument('--gold_tree_path', default=None, type=str)
    options = cmd.parse_args()
    
    predictor = TreeFileWrapper(options.tree_path)
    
    evaluator = ConstituentsEval(options.gold_tree_path, ['NNP', 'VP', 'NP', 'SBAR', 'S', 'ADJP'])
    evaluator.eval(predictor)