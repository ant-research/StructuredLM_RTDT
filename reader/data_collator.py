# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

from functools import partial
import torch
from typing import List, Dict
from reader.memory_line_reader import InputItem
import numpy as np
import random
from utils.tree_utils import build_mlm_inputs


class DefaultCollator:
    def __init__(self, tokenizer, include_atom_span=False) -> None:
        self._tokenizer = tokenizer
        self._include_atom_span = include_atom_span
    
    def default_data_collator(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        result = {"input_ids": torch.tensor(np.array(input_ids_batch)), "attention_mask": torch.tensor(np.array(mask_batch))}
        if self._include_atom_span:
            result['atom_spans'] = [item.atom_spans for item in items[0]]
        return result
    
    def default_fastr2d2_collator(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        model_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "masks": torch.tensor(np.array(mask_batch), dtype=torch.long)
        }
        parser_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)
        }
        if self._include_atom_span:
            model_inputs['atom_spans'] = [item.atom_spans for item in items[0]]
        return model_inputs, parser_inputs
    
    def default_fastr2d2_nsp_collator(self, items):
        ids_batch = []
        is_next_sent_batch = []
        for sent_a, sent_b, is_next_sent in items[0]:
            ids_batch.append(sent_a.ids)
            ids_batch.append(sent_b.ids)
            is_next_sent_batch.append(is_next_sent)
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        
        input_ids_batch = []
        mask_batch = []
        
        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
        
        model_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                        "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                        "pairwise":True, "pair_targets": torch.tensor(np.array(is_next_sent_batch))}
        parser_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                         "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
        return model_inputs, parser_inputs
    
    def default_fastr2d2_nsp_mlm_collator(self, items):
        ids_batch = []
        is_next_sent_batch = []
        for sent_a, sent_b, is_next_sent in items[0]:
            ids_batch.append(sent_a.ids)
            ids_batch.append(sent_b.ids)
            is_next_sent_batch.append(is_next_sent)
        lens = [len(a) for a in ids_batch]
        input_max_len = max(1, max(lens))
        
        input_ids_batch = []
        mask_batch = []
        
        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
        
        model_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                        "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                        "tree_to_sequence": partial(build_mlm_inputs, input_ids_np = np.array(input_ids_batch), seq_lens_np=np.array(lens), 
                                                    pairwise=True, mask_roller=lambda x: np.random.rand() <= 0.15), 
                        "pair_targets": torch.tensor(np.array(is_next_sent_batch))}
        parser_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                         "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
        return model_inputs, parser_inputs

    def default_fastr2d2_mlm_collator(self, items):
        ids_batch = []
        is_next_sent_batch = []
        for sent_a, is_next_sent in items[0]:
            ids_batch.append(sent_a.ids)
            is_next_sent_batch.append(is_next_sent)
        lens = [len(a) for a in ids_batch]
        input_max_len = max(1, max(lens))
        
        input_ids_batch = []
        mask_batch = []
        
        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
        
        model_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                        "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                        "tree_to_sequence": partial(build_mlm_inputs, input_ids_np = np.array(input_ids_batch), seq_lens_np=np.array(lens), 
                                                    pairwise=False, mask_roller=lambda x: np.random.rand() <= 0.15)}
        parser_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                         "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
        return model_inputs, parser_inputs


class MLMDataCollator:
    def __init__(self, tokenizer, mlm_rate=0.15, mlm_replace_rate=0.2) -> None:
        self._tokenizer = tokenizer
        self._mlm_rate = mlm_rate
        self._mlm_replace_rate = mlm_replace_rate

    def vanilla_mlm_data_collator(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            for idx, input_id in enumerate(masked_input_ids):
                if np.random.rand() < self._mlm_rate:
                    target_ids[idx] = input_id
                    if np.random.rand() < self._mlm_replace_rate:
                        masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                    else:
                        masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.bool),
                "target_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long)}

    def fastr2d2_cio_mlm_atom_span_collator(self, items):
        ids_batch = [item.ids for item in items[0]]
        atom_spans = [item.atom_spans for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        org_input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            org_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            for idx, input_id in enumerate(masked_input_ids):
                if np.random.rand() < self._mlm_rate:
                    target_ids[idx] = input_id
                    if np.random.rand() < self._mlm_replace_rate:
                        masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                    else:
                        masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            org_input_ids = np.append(org_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            org_input_ids_batch.append(org_input_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))
        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "pairwise":False,
                "atom_spans": atom_spans,
                "tgt_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long)},\
               {"input_ids": torch.tensor(np.array(org_input_ids_batch), dtype=torch.long),
                "atom_spans": atom_spans,
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
        

    def fast_r2d2_mlm_data_collator(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        org_input_ids_batch = []
        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            for idx, input_id in enumerate(masked_input_ids):
                if np.random.rand() < self._mlm_rate:
                    target_ids[idx] = input_id
                    if np.random.rand() < self._mlm_replace_rate:
                        masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                    else:
                        masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_org_ids = np.append(input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            org_input_ids_batch.append(padded_org_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "target_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long)},\
               {"input_ids": torch.tensor(np.array(org_input_ids_batch), dtype=torch.long),
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
               
    def fastr2d2_cio_nsp_mlm_collator(self, items):
        ids_batch = []
        is_next_sent_batch = []
        for sent_a, sent_b, is_next_sent in items[0]:
            ids_batch.append(sent_a.ids)
            ids_batch.append(sent_b.ids)
            is_next_sent_batch.append(is_next_sent)
        lens = [len(a) for a in ids_batch]
        input_max_len = max(1, max(lens))
        
        input_ids_batch = []
        masked_ids_batch = []
        tgt_ids_batch = []
        mask_batch = []
        
        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            rand_vals = np.random.rand(len(input_ids))
            masked_pos = list(filter(lambda x: rand_vals[x] < self._mlm_rate, range(len(input_ids))))
            for idx in masked_pos:
                target_ids[idx] = input_ids[idx]
                if np.random.rand() < self._mlm_replace_rate:
                    masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                else:
                    masked_input_ids[idx] = self._tokenizer.mask_token_id
                        
            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            masked_ids_batch.append(padded_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            tgt_ids_batch.append(padded_tgt_ids)
                        
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
        
        model_inputs = {"input_ids": torch.tensor(np.array(masked_ids_batch), dtype=torch.long), 
                        "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                        "pairwise":True,
                        "tgt_ids": torch.tensor(tgt_ids_batch, dtype=torch.long),
                        "pair_targets": torch.tensor(np.array(is_next_sent_batch))}
        parser_inputs = {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                         "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
        return model_inputs, parser_inputs
    
    def fastr2d2_cio_mlm_collator(self, items):
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        org_input_ids_batch = []
        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            rand_vals = np.random.rand(len(input_ids))
            masked_pos = list(filter(lambda x: rand_vals[x] < self._mlm_rate, range(len(input_ids))))
            for idx in masked_pos:
                target_ids[idx] = input_ids[idx]
                if np.random.rand() < self._mlm_replace_rate:
                    masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                else:
                    masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_org_ids = np.append(input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            org_input_ids_batch.append(padded_org_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "pairwise":False,
                "tgt_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long)},\
               {"input_ids": torch.tensor(np.array(org_input_ids_batch), dtype=torch.long),
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long)}
               
    def fastr2d2_cio_mlm_pairwise_collator(self, items):
        flatten_items = []
        for item_a, item_b in items[0]:
            flatten_items.append(item_a.ids)
            flatten_items.append(item_b.ids)
        results = self.fastr2d2_cio_mlm_collator(flatten_items)
        results['pairwise'] = True
        return results
               
class GlueCollator:
    def __init__(self, data_type, mask_id, sep_id, mask_rate=0.0, mask_epochs=0) -> None:
        self._data_type = data_type
        self._mask_rate = mask_rate
        self._mask_id = mask_id
        self._sep_id = sep_id
        # self._decline_rate = decline_rate
        self._mask_epochs = mask_epochs
        self._epoch = 0
    
    def set_epoch(self, epoch):
        self._epoch = epoch
        
    def _get_mask_rate(self):
        if self._epoch < self._mask_epochs:
            return self._mask_rate
        return 0.0
        
    def mlm_concat_collator(self, items):
        assert len(items) == 1
        input_items = items[0]
        if self._data_type == 'pair':
            input_items = list(filter(lambda x: len(x.ids_sep[0]) + len(x.ids_sep[1]) < 512, input_items))
            lens = map(lambda x: len(x.ids_sep[0]) + len(x.ids_sep[1]), input_items)
        else:
            lens = map(lambda x: len(x.ids), input_items)
        # input_max_len = max(1, max(lens)) + 1
        input_max_len = max(1, max(lens))  # for no sep-id

        input_ids_batch, tgt_ids_batch, mask_batch, labels_batch = [], [], [], []
        sep_id = self._sep_id
        for input_item in input_items:
            if self._data_type == 'pair':
                ids_a, ids_b = input_item.ids_sep
                # ids_a, ids_b = torch.tensor(ids_a), torch.tensor(ids_b)
                # concat_ids = torch.tensor(ids_a + [sep_id] + ids_b)
                concat_ids = torch.tensor(ids_a + ids_b)
                tgt = torch.zeros((input_max_len,), dtype=torch.long).fill_(-1)
                input_ids = torch.zeros((input_max_len,), dtype=torch.long)
                mask_rate = self._get_mask_rate()
                mask_len = round(concat_ids.shape[0] * mask_rate)
                if mask_len > 0:
                    mask_ids = random.sample(range(concat_ids.shape[0]), mask_len)
                    tgt[mask_ids] = concat_ids[mask_ids]
                    concat_ids[mask_ids] = self._mask_id
                    
                label_idx = input_item.label

                padding_len = input_max_len - concat_ids.shape[0]
                tgt_ids_batch.append(tgt)
                input_ids[:concat_ids.shape[0]] = concat_ids
                input_ids_batch.append(input_ids)
                mask_batch.append([1] * concat_ids.shape[0] + [0] * padding_len)

                labels_batch.append(label_idx)
            else:
                ids = torch.tensor(input_item.ids)
                org_ids = torch.tensor(input_item.ids)
                tgt = torch.zeros((input_max_len,), dtype=torch.long).fill_(-1)
                mask_rate = self._get_mask_rate()
                mask_len = round(len(ids) * mask_rate)
                if mask_len > 0:
                    mask_ids = random.sample(range(len(ids)), mask_len)
                    tgt[mask_ids] = ids[mask_ids]
                    ids[mask_ids] = self._mask_id
                label_idx = input_item.label
                padding_len = input_max_len - len(ids)
                input_ids_tensor = torch.cat((ids, torch.tensor([0]*padding_len, dtype=ids.dtype)))
                input_ids_batch.append(input_ids_tensor)
                tgt_ids_batch.append(tgt)
                mask_batch.append([1] * len(ids) + [0] * padding_len)
                labels_batch.append(label_idx)
        kw_item = {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "tgt_ids": torch.stack(tgt_ids_batch, dim=0),
            "attention_mask": torch.tensor(mask_batch),
            "labels": (torch.tensor(labels_batch, dtype=torch.long)),
        }
        return kw_item
    
    def mlm_collator(self, items):
        assert len(items) == 1
        input_items = items[0]
        if self._data_type == 'pair':
            lens = map(lambda x: max(len(x.ids_sep[0]), len(x.ids_sep[1])), input_items)
        else:
            lens = map(lambda x: len(x.ids), input_items)
        input_max_len = max(1, max(lens))

        input_ids_batch, tgt_ids_batch, mask_batch, labels_batch = [], [], [], []
        org_ids_batch = []
        for input_item in input_items:
            if self._data_type == 'pair':
                ids_a, ids_b = input_item.ids_sep
                org_ids_a, org_ids_b = torch.tensor(ids_a), torch.tensor(ids_b)
                ids_a, ids_b = torch.tensor(ids_a), torch.tensor(ids_b)
                tgt_a = torch.zeros((input_max_len,), dtype=torch.long).fill_(-1)
                tgt_b = torch.zeros((input_max_len,), dtype=torch.long).fill_(-1)
                mask_rate = self._get_mask_rate()
                mask_len_a = round(len(ids_a) * mask_rate)
                mask_len_b = round(len(ids_b) * mask_rate)
                if mask_len_a > 0:
                    mask_ids_a = random.sample(range(len(ids_a)), mask_len_a)
                    tgt_a[mask_ids_a] = ids_a[mask_ids_a]
                    ids_a[mask_ids_a] = self._mask_id
                if mask_len_b > 0:
                    mask_ids_b = random.sample(range(len(ids_b)), mask_len_b)
                    tgt_b[mask_ids_b] = ids_b[mask_ids_b]
                    ids_b[mask_ids_b] = self._mask_id
                    
                label_idx = input_item.label

                padding_len_a = input_max_len - len(ids_a)
                padding_len_b = input_max_len - len(ids_b)
                tgt_ids_batch.append(tgt_a)
                tgt_ids_batch.append(tgt_b)
                input_ids_batch.append(torch.cat((ids_a,torch.tensor([0] * padding_len_a, dtype=ids_a.dtype))))
                input_ids_batch.append(torch.cat((ids_b,torch.tensor([0] * padding_len_b, dtype=ids_b.dtype))))
                org_ids_batch.append(torch.cat((org_ids_a,torch.tensor([0] * padding_len_a, dtype=ids_a.dtype))))
                org_ids_batch.append(torch.cat((org_ids_b,torch.tensor([0] * padding_len_b, dtype=ids_b.dtype))))
                mask_batch.append([1] * len(ids_a) + [0] * padding_len_a)
                mask_batch.append([1] * len(ids_b) + [0] * padding_len_b)

                labels_batch.append(label_idx)
            else:
                ids = torch.tensor(input_item.ids)
                org_ids = torch.tensor(input_item.ids)
                tgt = torch.zeros((input_max_len,), dtype=torch.long).fill_(-1)
                mask_rate = self._get_mask_rate()
                mask_len = round(len(ids) * mask_rate)
                if mask_len > 0:
                    mask_ids = random.sample(range(len(ids)), mask_len)
                    tgt[mask_ids] = ids[mask_ids]
                    ids[mask_ids] = self._mask_id
                label_idx = input_item.label
                padding_len = input_max_len - len(ids)
                input_ids_tensor = torch.cat((ids, torch.tensor([0]*padding_len, dtype=ids.dtype)))
                input_ids_batch.append(input_ids_tensor)
                tgt_ids_batch.append(tgt)
                org_ids_batch.append(torch.cat((org_ids, torch.tensor([0]*padding_len, dtype=ids.dtype))))
                mask_batch.append([1] * len(ids) + [0] * padding_len)
                labels_batch.append(label_idx)
        kw_item = {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "parser_ids": torch.stack(org_ids_batch, dim=0),
            "tgt_ids": torch.stack(tgt_ids_batch, dim=0),
            "pairwise": self._data_type == 'pair',
            "attention_mask": torch.tensor(mask_batch),
            "labels": (torch.tensor(labels_batch, dtype=torch.long)),
        }
        return kw_item
    
class TransformerPretarinCollator:
    def __init__(self, tokenizer, mlm_rate=0.15, mlm_replace_rate=0.1) -> None:
        self._mlm_rate = mlm_rate
        self._mlm_replace_rate = mlm_replace_rate
        self._tokenizer = tokenizer
    
    def chunked_mlm_collator(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        sep_id = self._tokenizer.get_vocab()['[SEP]']
        ids_batch = []
        for chunk in items[0]:
            ids_chunk = []
            for input_item, _ in chunk:
                ids_chunk.extend(input_item.ids)
                ids_chunk.append(sep_id)
            ids_batch.append(ids_chunk)
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            for idx, input_id in enumerate(masked_input_ids):
                if np.random.rand() < self._mlm_rate:
                    target_ids[idx] = input_id
                    if np.random.rand() < self._mlm_replace_rate:
                        masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                    else:
                        masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.bool),
                "target_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long)}