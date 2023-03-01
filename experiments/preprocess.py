# coding=utf-8
# Copyright (c) 2021 Ant Group

from supar import Parser
import codecs
from nltk.tree import Tree
import json
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from data_structure.const_tree import SpanTree
from data_structure.r2d2_tree import PyNode


# TODO: load raw text from ATIS or other dataset and then call build_trees to build str-> Tree mappings.
# When loading ATIS corpus, load str->Tree mapping and convert to span informations.


def atis_clean():
    data_path = 'data/ATIS/standard_format/rasa/train.json'
    data_path_test = 'data/ATIS/standard_format/rasa/test.json'
    output_path = 'data/ATIS/raw_cleaned.txt'
    # raw_data = []
    with open(data_path,'r') as f, open(data_path_test,'r') as f_test, open(output_path, 'w') as f_out:
        data = json.load(f)
        data_test = json.load(f_test)
        for line in data['rasa_nlu_data']['common_examples']:
            f_out.write(line['text'] + '\n')
        for line in data_test['rasa_nlu_data']['common_examples']:
            f_out.write(line['text'] + '\n')

def build_trees(corpus, output_path, parser_path):
    parser = Parser.load(parser_path)
    with codecs.open(corpus, mode='r', encoding='utf-8') as f_in, \
        codecs.open(output_path, mode='w', encoding='utf-8') as f_out:
        for _line in tqdm(f_in):
            if len(_line.strip()) > 0:
                tokens = _line.split()
                try:
                    dataset = parser.predict(tokens, lang=None, verbose=False)
                    print(f'\0{_line.strip()}', file=f_out)
                    dataset.trees[0].pprint(stream=f_out)
                except:
                    print(_line)


def load_trees(corpus):
    def build_tree_from_lines(lines):
        tree_expr = '\n'.join(lines)
        tree_expr = tree_expr.replace('_ (', '_ -LBR-').replace('_ )', '_ -RBR-')
        return Tree.fromstring(tree_expr)

    str_tree_map = {}
    with codecs.open(corpus, mode='r', encoding='utf-8') as f_in:
        line_buffer = []
        current_utterance = None
        for _line in tqdm(f_in):
            if _line.startswith('\0'):
                if len(line_buffer) > 0:
                    str_tree_map[current_utterance] = build_tree_from_lines(line_buffer)
                    span_tree_check(convert_tree_to_span(str_tree_map[current_utterance]))
                    line_buffer.clear()
                current_utterance = _line[1:].strip()
            else:
                line_buffer.append(_line)
        
        if len(line_buffer) > 0:
            str_tree_map[current_utterance] = build_tree_from_lines(line_buffer)
            span_tree_check(convert_tree_to_span(str_tree_map[current_utterance]))
    return str_tree_map


def convert_tree_to_span(tree:Tree, indices_mapping=None, offset=0):
    span_length = 0
    sub_spans = []
    current_offset = offset
    while isinstance(tree, Tree) and len(tree) == 1:
        tree = tree[0]
    if isinstance(tree, Tree):
        for sub_tree in tree:
            if isinstance(sub_tree, Tree):
                sub_span_node = convert_tree_to_span(sub_tree, indices_mapping, current_offset)
                sub_spans.append(sub_span_node)
                if indices_mapping is not None:
                    for cur_offset, (st, _) in enumerate(indices_mapping):
                        if st == sub_span_node.ed + 1:
                            current_offset = cur_offset
                            break
                else:
                    current_offset = sub_span_node.ed + 1
                span_length += sub_span_node.ed - sub_span_node.st + 1
            else:
                if indices_mapping is not None:
                    span_length += 1 + indices_mapping[offset][1] - indices_mapping[offset][0]
                else:
                    span_length += 1
    else:
        if indices_mapping is not None:
            span_length = 1 + indices_mapping[offset][1] - indices_mapping[offset][0]
        else:
            span_length = 1
    span_st = indices_mapping[offset][0] if indices_mapping is not None else offset
    span_node = SpanTree(span_st, span_st + span_length - 1, sub_spans)
    return span_node


def convert_tree_to_node(tree:SpanTree, right_pointer=-1):
    if len(tree.subtrees) == 0:
        return generate_pynode(tree.st, tree.ed)
    if right_pointer == -1:
        right_pointer = len(tree.subtrees) - 1
    right_subtrees = tree.subtrees[right_pointer]
    right_node = convert_tree_to_node(right_subtrees, -1)
    if right_pointer > 1:
        right_pointer -= 1
        left_node = convert_tree_to_node(tree, right_pointer)
    else:
        left_subtrees = tree.subtrees[0]
        left_node = convert_tree_to_node(left_subtrees, -1)
    current_node = PyNode(left_node, right_node, left_node.i, right_node.j, None)
    return current_node


def generate_pynode(st, ed):
    if st == ed:
        return PyNode(None, None, st, ed, None)
    right_node = generate_pynode(ed, ed)
    left_node = generate_pynode(st, ed - 1)
    current_node = PyNode(left_node, right_node, st, ed, None)
    return current_node

def span_tree_check(tree:SpanTree, offset=-1):
    if len(tree.subtrees) == 0:
        assert tree.st <= tree.ed
        return 
    
    for subtree in tree.subtrees:
        assert subtree.st <= subtree.ed
        assert subtree.st == offset+1
        span_tree_check(subtree, offset)
        offset = subtree.ed


def extract_tsv_raw_text(tsv_path, column_idx, output_path):
    dataframe = pd.read_csv(tsv_path, sep='\t', header=0)
    with codecs.open(output_path, mode='w', encoding='utf-8') as f_out:
        for row in dataframe.values:
            print(row[column_idx], file=f_out)


if __name__ =='__main__':
    # pass
    extract_tsv_raw_text('data/key_word_mining/voice_tmp.csv', 0, 'data/key_word_mining/raw_text.txt')
    # extract_tsv_raw_text('data/glue/SST-2/train.tsv', 0, 'data/glue/SST-2/train.raw.txt')
    # build_trees('data/glue/SST-2/dev.raw.txt', 'data/glue/SST-2/dev.trees.txt',\
    #             'data/const_parser/ptb.crf.constituency.char')

    # extract_tsv_raw_text('data/glue/CoLA/dev.tsv', 3, 'data/glue/CoLA/dev.raw.txt')
    # build_trees('data/glue/CoLA/dev.raw.txt', 'data/glue/CoLA/dev.trees.txt',\
    #             'data/const_parser/ptb.crf.constituency.char')
    # build_trees('data/stanfordLU/weather_raw_data', 'data/stanfordLU/weather_trees.txt',\
    #             'data/const_parser/ptb.crf.constituency.char')