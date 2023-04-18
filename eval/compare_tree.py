#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
import numpy as np
import time
from utils.tree_utils import get_nonbinary_spans, get_stats
import re

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--tree1', default='')
parser.add_argument('--tree2', default='')
parser.add_argument('--length_cutoff', default=150, type=int)


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not (char == '(')
        output.append(char)
    return ''.join(output)


def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not (
        is_next_open_bracket(line_strip, i)):  # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    # print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2  # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '('  # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        if (char == '(') and not (char == ')'):
            raise Exception('bracket error')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i):  # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                while line_strip[
                    i] != '(':  # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else:  # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
            output_actions.append('REDUCE')
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ')' and line_strip[i] != '(':
                i += 1
    assert i == max_idx
    return output_actions

def get_level(val, division):
    final_level = -1
    for level, depth_threshold in enumerate(division):
        if val <= depth_threshold:
            final_level = level
            break
    if final_level == -1:
        final_level = len(division)
    return final_level

def main(args):
    corpus_f1 = [0., 0., 0.]
    corpus_len = {}
    sent_f1 = []
    sent_len = {}
    len_div = [10, 20, 40]
    gold_tree_cache = {}
    for tree in open(args.tree1, 'r', encoding='utf-8'):
        try:
            gold_tree = tree.strip()
            action = get_actions(gold_tree)
            tags1, sent1, sent_lower1 = get_tags_tokens_lowercase(tree)
            gold_tree_cache[' '.join(sent_lower1)] = action
        except:
            print(tree)

    with torch.no_grad():
        # for k, (tree1, tree2) in enumerate(zip(open(args.tree1, "r", encoding='utf-8'),
        #                                        open(args.tree2, 'r', encoding='utf-8'))):
        for k, tree2 in enumerate(open(args.tree2, 'r', encoding='utf-8')):
            # tree1 = tree1.strip()
            # action1 = get_actions(tree1)
            # tags1, sent1, sent_lower1 = get_tags_tokens_lowercase(tree1)
            # if len(sent1) > args.length_cutoff or len(sent1) == 1:
            #     continue
            # gold_span1, binary_actions1, nonbinary_actions1 = get_nonbinary_spans(action1)
            tree2 = tree2.strip()
            action2 = get_actions(tree2)
            tags2, sent2, sent_lower2 = get_tags_tokens_lowercase(tree2)
            if u' '.join(sent_lower2) not in gold_tree_cache:
                print(' '.join(sent2))
                continue
            action1 = gold_tree_cache[u' '.join(sent_lower2)]
            gold_span1, binary_actions1, nonbinary_actions1 = get_nonbinary_spans(action1)
            len_level = get_level(len(sent2), len_div)
            # assert len(sent1) == len(sent2)
            # assert u' '.join(sent1).lower() == u' '.join(sent2).lower()
            gold_span2, binary_actions2, nonbinary_actions2 = get_nonbinary_spans(action2)
            pred_span_set = set(gold_span2[:-1])  # the last span in the list is always the
            gold_span_set = set(gold_span1[:-1])  # trival sent-level span so we ignore it
            tp, fp, fn = get_stats(pred_span_set, gold_span_set)
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            corpus_len.setdefault(len_level, [0.0, 0.0, 0.0])
            corpus_len[len_level][0] += tp
            corpus_len[len_level][1] += fp
            corpus_len[len_level][2] += fn
            # Sentence-level F1 is based on the original code from PRPN, i.e.
            # L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
            # As pointed out in the README, this code isn't entirely correct since sentence-level F1 could be
            # nonzero for short sentences with only a single sentence-level span.
            # In practice this discrepancy has minimal impact on the final results, but should ideally be
            # fixed nonetheless.
            overlap = pred_span_set.intersection(gold_span_set)
            prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
            if len(gold_span_set) == 0:
                reca = 1.
                if len(pred_span_set) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)
            sent_len.setdefault(len_level, [])
            sent_len[len_level].append(f1)
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    print('Corpus F1: %.2f, Sentence F1: %.2f' %
          (corpus_f1 * 100, np.mean(np.array(sent_f1)) * 100))
    for level, (tp, fp, fn) in corpus_len.items():
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        print(f'Corpus F1 len {level}: {corpus_f1}')
    for level, f1_list in sent_len.items():
        print(f'Sentence F1 len {level}: {np.mean(np.array(f1_list))}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
