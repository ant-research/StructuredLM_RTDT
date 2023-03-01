# coding=utf-8
# Copyright (c) 2021 Ant Group

import codecs
from data_structure.basic_structure import DotDict
from data_structure.syntax_tree import ConllTree, ConllNode
import os

COMMONT_PREFIX = '#'


class ConllReader(object):
    def __init__(self, sample_mode, max_len, logger):
        self.sample_mode = sample_mode
        self.max_len = max_len
        self.logger = logger

    def create_sample(self, buffer):
        need_retokenize = True
        parents = postags = relations = all_columns = None
        if self.sample_mode == 'tree':
            try:
                all_columns = [[] for _ in range(10)]
                for line in buffer:
                    row = line.split()
                    assert len(row) == 10
                    for index, value in enumerate(row[:10]):
                        all_columns[index].append(value)
            except Exception as err:
                all_columns = None
                self.logger.exception('%s', err)

            tokens, parents, postags, relations = [], [-1], [''], ['ROOT']
            for line in buffer:
                parts = line.split()
                try:
                    parent_id = int(parts[6])
                except Exception as err:
                    parent_id = -1
                    self.logger.exception('%s', err)

                if parent_id < 0:
                    parent_id = -1
                tokens.append(parts[1])
                postags.append(parts[3])
                parents.append(parent_id)
                relations.append(parts[7])
        elif self.sample_mode == 'text':
            if isinstance(buffer, list):
                tokens = [line.split()[1] for line in buffer]
            else:
                need_retokenize = False
                tokens = self._tokenize(buffer)[0]

            all_columns = []
            all_columns.append([str(_ + 1) for _ in range(len(tokens))])
            all_columns.append(tokens)
            all_columns.extend(['_'] * len(tokens) for _ in range(8))

        if parents is None:
            parents = [-1] * (len(tokens) + 1)

        return DotDict(words=tokens,
                       parents=parents,
                       relations=relations,
                       postags=postags,
                       need_retokenize=need_retokenize,
                       all_columns=all_columns)

    def _filter_corpus(self, raw_data):
        for comments, data in raw_data:
            sample = self.create_sample(data)
            sample.comments = ''.join(comments)

            if len(sample.words) > self.max_len > 0:
                continue

            yield sample

    def from_conll_file(self, conll_path):
        buffers = []
        buffer = []
        comments = []
        with codecs.open(conll_path, mode='r', encoding='utf-8') as fp:
            for line in fp:
                if line.startswith(COMMONT_PREFIX):
                    comments.append(line)
                    continue
                line = line.strip()
                if not line:
                    if buffer and len(buffer) > 1:
                        buffers.append((comments, buffer))
                    if len(buffers) > 100:
                        break

                    comments = []
                    buffer = []
                else:
                    buffer.append(line)

            if buffer:
                buffers.append((comments, buffer))

        return list(self._filter_corpus(buffers))


def filter_conll_sentence(path, max_len, punct_set, is_CN=False):
    conll_sentences = []
    file_list = []
    if os.path.isdir(path):
        for root, dir, files in os.walk(path):
            for file in files:
                if file.find('conllu') >= 0 and file.find('train') >= 0:
                    file_list.append(os.path.join(root, file))
    else:
        file_list.append(path)
    for file_path in file_list:
        with codecs.open(file_path, mode='r', encoding='utf-8') as fin:
            tree_name = file_path.replace('train.conllu', '')
            buffer = []
            for line in fin:
                if line.startswith('#'):
                    continue
                if len(line.strip()) == 0:
                    total_len = len(buffer)
                    if not (total_len > max_len > 0):
                        if len(buffer) > 1:
                            conll_sentences.append(ConllTree(buffer, tree_name))
                    # if len(conll_sentences) > 10:
                    #     break
                    buffer = []
                    continue
                parts = line.strip().split()
                try:
                    if len(parts) == 5:
                        node = ConllNode(int(parts[0]), int(parts[3]), parts[1], parts[2], parts[4])
                    else:
                        node = ConllNode(int(parts[0]), int(parts[6]), parts[1], parts[3], parts[7])
                    buffer.append(node)
                except:
                    continue
    return conll_sentences