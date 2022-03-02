# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch
from torch.utils.data import Dataset
from transformers.data.processors import glue_processors, glue_output_modes
import numpy as np
import random
from typing import Dict


class R2D2GlueReader(Dataset):
    def __init__(self, task_name, data_dir, mode, tokenizer, max_batch_len, max_batch_size, batch_by_len=True):
        self.max_batch_len = max_batch_len
        self.max_batch_size = max_batch_size
        glue_processor = glue_processors[task_name]()
        if mode == "train":
            self.input_examples = glue_processor.get_train_examples(data_dir)
        elif mode == "dev":
            self.input_examples = glue_processor.get_dev_examples(data_dir)
        self.labels = glue_processor.get_labels()
        self.output_mode = glue_output_modes[task_name]
        self.mode = mode
        self.tokenizer = tokenizer
        self.batch_by_len = batch_by_len

        self.items = []
        self.model_type = "single"
        if self.input_examples[0].text_b is not None:
            self.model_type = "pair"
        for input_example in self.input_examples:
            current_item = []
            tokens = tokenizer.tokenize(input_example.text_a)
            total_len = len(tokens)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            current_item.append(ids)
            if input_example.text_b is not None:
                tokens = tokenizer.tokenize(input_example.text_b)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                current_item.append(ids)
                total_len += len(tokens)
            if self.output_mode == "classification":
                label_idx = self.labels.index(input_example.label)
                current_item.append(label_idx)
            elif self.output_mode == "regression":
                raise Exception("Regression not supported")
            else:
                raise Exception("Illegal output mode")
            if (mode == "train" and total_len < max_batch_len) or mode == "dev":
                self.items.append(current_item)

        print(f"Total number of examples {len(self.items)}")
        self._batches = self.batchify()

    def batchify(self):
        batches = []

        if self.batch_by_len:
            len_dict = {}
            for group in self.items:
                if len(group) == 3:
                    total_len = len(group[0]) + len(group[1])
                else:
                    total_len = len(group[0])
                arr = len_dict.get(total_len, [])
                arr.append(group)
                len_dict[total_len] = arr
            len_keys = list(len_dict.keys())
            len_keys.sort(key=lambda x: x, reverse=True)
            rest_lines = len(self.items)
            while rest_lines > 0:
                rest_len = self.max_batch_len
                current_batch = []
                while rest_len > 0 and len(current_batch) < self.max_batch_size:
                    next_len = -1
                    for key_len in len_keys:
                        if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                            next_len = key_len
                            break
                    if next_len != -1:
                        assert len(len_dict) > 0
                        group = len_dict[next_len].pop()
                        current_batch.append(group)
                        rest_len -= next_len
                        rest_lines -= 1
                    else:
                        break
                if len(current_batch) == 0:
                    # no sentence to add
                    break
                batches.append(current_batch)
            print(f"Total exmaples added: {len(self.items) - rest_lines}, skipped: {rest_lines}")
            print(f"Total # of batches: {len(batches)}")
        else:
            current_batch = []
            current_len = 0
            for item in self.items:
                if len(item) == 3:
                    ids_len = len(item[0]) + len(item[1])
                else:
                    ids_len = len(item[0])
                if current_len + ids_len > self.max_batch_len or len(current_batch) >= self.max_batch_size:
                    batches.append(current_batch)
                    current_batch = [item]
                    current_len = ids_len
                else:
                    current_batch.append(item)
                    current_len += ids_len
            if len(current_batch) > 0:
                batches.append(current_batch)
        return batches

    def shuffle(self):
        random.shuffle(self.items)
        self._batches = self.batchify()

    def get_output_mode(self):
        return self.output_mode

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, index):
        return self._batches[index]

    def collate_batch(self, ids_batch) -> Dict[str, torch.Tensor]:
        assert len(ids_batch) == 1
        groups = ids_batch[0]
        if len(groups[0]) == 3:
            lens = map(lambda x: max(len(x[0]), len(x[1])), groups)
        else:
            lens = map(lambda x: len(x[0]), groups)
        input_max_len = max(1, max(lens))

        input_ids_batch, mask_batch, labels_batch = [], [], []
        for group in groups:
            if len(group) == 3:
                text_a, text_b, labels = group

                padding_len_a = input_max_len - len(text_a)
                padding_len_b = input_max_len - len(text_b)
                input_ids_batch.append([text_a + [0] * padding_len_a, text_b + [0] * padding_len_b])
                mask_batch.append([[1] * len(text_a) + [0] * padding_len_a, 
                                   [1] * len(text_b) + [0] * padding_len_b])

                labels_batch.append(labels)

            elif len(group) == 2:
                text_a, labels = group
                padding_len_a = input_max_len - len(text_a)
                input_ids_batch.append(text_a + [0] * padding_len_a)
                mask_batch.append([1] * len(text_a) + [0] * padding_len_a)

                labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch),
            "attention_mask": torch.tensor(mask_batch),
            "labels": torch.tensor(labels_batch, dtype=torch.long)
            if self.output_mode == "classification"
            else torch.tensor(labels_batch, dtype=torch.float),
        }
