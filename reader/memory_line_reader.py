# coding=utf-8
# Copyright (c) 2021 Ant Group

from torch.utils.data import Dataset
import torch
import codecs
import copy
from typing import List, Dict
import random
from utils.misc import _align_spans, get_sentence_from_words
from utils.treebank_processor import get_tags_tokens_lowercase
import numpy as np
import logging


class InputItem:
    def __init__(self, ids, atom_spans=None) -> None:
        self.ids = ids
        self.atom_spans = atom_spans


class BatchByLengthDataset(Dataset):
    def __init__(self, max_batch_size, max_batch_len, random) -> None:
        super().__init__()
        self._lines = []
        self._batches = []
        self._max_batch_size = max_batch_size
        self._max_batch_len = max_batch_len
        self._random = random

    def shuffle(self):
        random.shuffle(self._lines)
        self._batches = self.batchify()

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        return self._batches[idx]

    def collate_batch(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [0] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        return {"input_ids": torch.tensor(input_ids_batch), "attention_mask": torch.tensor(mask_batch),
                "atom_spans": [item.atom_spans for item in items[0]]}

    def batchify(self):
        if not self._random:
            logging.info("batchify")
            len_dict = {}
            for input_item in self._lines:
                arr = len_dict.get(len(input_item.ids), [])
                arr.append(input_item)
                len_dict[len(input_item.ids)] = arr
            len_keys = list(len_dict.keys())
            len_keys.sort(key=lambda x: x, reverse=True)
            rest_lines = len(self._lines)
            batches = []
            while rest_lines > 0:
                rest_len = self._max_batch_len
                current_batch = []
                while rest_len > 0 and len(current_batch) < self._max_batch_size:
                    next_len = -1
                    for key_len in len_keys:
                        if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                            next_len = key_len
                            break
                    if next_len != -1:
                        assert len(len_dict) > 0
                        item = len_dict[next_len].pop()
                        current_batch.append(item)
                        rest_len -= next_len
                        rest_lines -= 1
                    else:
                        break
                if len(current_batch) == 0:
                    # no sentence to add
                    break
                batches.append(current_batch)
            return batches  # [_ for _ in reversed(batches)]
        else:
            logging.info("batchify")
            batches = []
            current_batch = []
            current_len_sum = 0
            for item in self._lines:
                if (current_len_sum + len(item.ids)) >= self._max_batch_len or \
                    len(current_batch) >= self._max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_len_sum = 0
                current_batch.append(item)
                current_len_sum += len(item.ids)
            if len(current_batch) > 0:
                batches.append(current_batch)
            return batches


class BatchSelfRegressionLineDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_line=1000, input_type="txt", random=False,
                 seperator=None):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(max_batch_size=batch_size, max_batch_len=batch_max_len, random=random)
        self._random = random
        self._tokenizer = tokenizer
        self._batch_max_len = batch_max_len
        self._batch_size = batch_size

        assert input_type in ["txt", "ids"]

        with codecs.open(path, mode="r", encoding="utf-8") as f:
            for _line in f:
                token_ids = None
                atom_spans = None
                if input_type == "txt":
                    if seperator is None:
                        tokens = self._tokenizer.tokenize(_line.strip())
                        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                    else:
                        try:
                            sentence, spans = get_sentence_from_words(_line.strip().split(seperator), seperator)
                            outputs = self._tokenizer.encode_plus(sentence,
                                                                  add_special_tokens=False,
                                                                  return_offsets_mapping=True)
                            new_spans = outputs['offset_mapping']
                            word_starts, word_ends = _align_spans(spans, new_spans)
                            atom_spans = []
                            for st, ed in zip(word_starts, word_ends):
                                if st != ed:
                                    atom_spans.append([st, ed])
                            token_ids = outputs['input_ids']
                            atom_spans = atom_spans
                        except Exception:
                            pass
                elif input_type == "ids":
                    parts = _line.strip().split('|')
                    token_ids = [int(t_id) for t_id in parts[0].split()]
                    tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
                    if len(parts) > 1:
                        spans = parts[1].split(';')
                        atom_spans = []
                        for span in spans:
                            vals = span.split(',')
                            if len(vals) == 2:
                                atom_spans.append([int(vals[0]), int(vals[1])])
                if min_len < len(token_ids) < self._batch_max_len:
                    self._lines.append(InputItem(token_ids, atom_spans))
                if len(self._lines) > max_line > 0:
                    break
        self.shuffle()
    
    def convert_ids_to_tokens(self, ids):
        return self._tokenizer.convert_ids_to_tokens(ids)


class TreeBankDataset(BatchByLengthDataset):
    def __init__(self, treebank_path, vocab_path, batch_max_len, batch_size,
                 random=False,) -> None:
        super().__init__(max_batch_size=batch_size, max_batch_len=batch_max_len, random=random)

        self.word2idx, self.idx2word = self.load_vocab(vocab_path)
        self._lines = []
        with codecs.open(treebank_path, mode='r', encoding='utf-8') as fin:
            for line in fin:
                _, _, lower_tokens = get_tags_tokens_lowercase(line)
                ids = [self.word2idx[t] if t in self.word2idx else self.word2idx['<unk>'] for t in lower_tokens]
                input_item = InputItem(ids, None)
                if len(ids) > 2:
                    self._lines.append(input_item)
        self.shuffle()

    def load_vocab(self, vocab_file):
        word2idx = {}
        idx2word = {}
        for line in open(vocab_file, 'r', encoding='utf-8'):
            v, k = line.strip().split()
            word2idx[v] = int(k)
        for word, idx in word2idx.items():
            idx2word[idx] = word
        return word2idx, idx2word

    def _convert(self, x):
        return torch.from_numpy(np.asarray(x))

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for w_id in ids.to('cpu').data.numpy():
            tokens.append(self.idx2word[w_id])
        return tokens