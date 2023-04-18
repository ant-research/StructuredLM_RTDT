from ast import Pass
from html import entities
import json
import codecs
from numpy import True_, mean
import torch
import os
import copy
import random
import sys
from reader.memory_line_reader import BatchByLengthDataset, InputItem
from data_structure.const_tree import SpanTree
from utils.misc import _align_spans, get_sentence_from_words
from experiments.preprocess import load_trees,convert_tree_to_span
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AutoTokenizer
from typing import List, Dict


class SLSReader(BatchByLengthDataset):
    """
    For the sls datasets
    """
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                    min_len=2, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        self.id2label_dict = {}
        self.label2id_dict = {}
        self.labels = []
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                        min_len, max_line, random, **kwargs)


    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        domain = kwargs['domain']
        mode = kwargs['mode']
        assert domain in ['movie_eng', 'movie_trivia10k13', 'restaurant']

        input_item_list = []
        domain = domain.split('_')
        
        with open(os.path.join(data_path_or_dir, '{}/{}_labels.txt'.format(domain[0],domain[-1])),'r') as f:
            for intent in f:
                intent = intent.strip()
                self.labels.append(intent)
                self.label2id_dict[intent] = len(self.label2id_dict)
                self.id2label_dict[len(self.id2label_dict)] = intent
        
        with codecs.open(os.path.join(data_path_or_dir, "{}/{}{}.bio".format(domain[0], domain[-1], mode)), 'r', encoding='utf-8') as f:
            words = []
            labels_all = []
            for line in f:
                line = line.strip().split('\t')
                if len(line) == 1:
                    sentence, spans = get_sentence_from_words(words, " ")
                    outputs = self._tokenizer.encode_plus(sentence,
                                                            add_special_tokens=False,
                                                            return_offsets_mapping=True)
                    new_spans = outputs['offset_mapping']
                    word_starts, word_ends = _align_spans(spans, new_spans)
                    atom_spans = []
                    for st, ed in zip(word_starts, word_ends):
                        if st != ed:
                            atom_spans.append((st, ed))
                    pre_label = ['O']
                    entities = []
                    entity = {"start":0, "end":0, "span_length":0}
                    span_length = 0
                    labels = []

                    for i, label in enumerate(labels_all):
                        label = label.split('-')
                        if label[0] != 'O' and self.label2id_dict[label[1]] not in labels:
                            labels.append(self.label2id_dict[label[1]])
                        if pre_label[0] != 'O' and pre_label[0] != label[0] and label[0] != 'I' and entity['start'] != 0:
                            entity['end'] = len(' '.join(words[:i]))
                            assert entity['end'] > entity['start'],'error'
                            entity['span_length'] = span_length
                            entities.append(entity)
                            entity = {"start":0, "end":0, "span_length":0}
                        if label[0] != 'O' and pre_label[0] != label[0] and entity['start'] == 0:
                            entity['entity'] = label[1]
                            entity['start'] = len(' '.join(words[:i])) + 1
                            span_length = 0
                            if entity['start'] == 1:
                                entity['start'] = 0
                        span_length += 1
                        pre_label = label
                    if label[0] != 'O' and entity['start'] != 0:
                        entity['end'] = len(' '.join(words[:i+1]))
                        assert entity['end'] > entity['start'],'error'
                        entity['span_length'] = span_length
                        entities.append(entity)
                    if len(entities) == 0:
                        mean_span_length = 0.
                    else:
                        mean_span_length = mean([en['span_length'] for en in entities])
                    for en in entities:
                        en['mean_span_length'] = mean_span_length
                    span_tree = None # convert_tree_to_span(self.str_tree_map[sentence])
                    token_ids = outputs['input_ids']
                    # tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
                    if self._min_len < len(token_ids) < self._batch_max_len:
                        input_item_list.append(InputItem(token_ids, atom_spans, labels=labels, span_tree=span_tree, \
                            sentence=sentence, offset_mapping=outputs['offset_mapping'], entities=entities, mean_span_length=mean_span_length))
                    sentence = []
                    labels = []
                    if len(input_item_list) > self._max_line > 0:
                        break
                    words = []
                    labels_all = []
                else:
                    words.append(line[1])
                    labels_all.append(line[0])
                    
        print(f"Total number of examples {len(input_item_list)}")
        return input_item_list


    def collate_batch(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        labels_batch = [item.kwargs['labels'] for item in items[0]]
        atom_spans = [item.atom_spans for item in items[0]]
        entities = [item.kwargs['entities'] for item in items[0]]
        offset_mapping = [item.kwargs['offset_mapping'] for item in items[0]]
        mean_span_length = [item.kwargs['mean_span_length'] for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []

        for input_ids, label_ids in zip(ids_batch, labels_batch):
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append(label_ids)
        
        batch_data = {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "atom_spans":atom_spans,
                "entities":entities,
                "offset_mapping":offset_mapping,
                "mean_span_length":mean_span_length,
                "labels":input_labels_batch}
        return batch_data
    
    def collate_batch_bert(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [[self._tokenizer.cls_token_id] + item.ids + [self._tokenizer.sep_token_id] for item in items[0]]
        labels_batch = [item.kwargs['labels'] for item in items[0]]
        trees_batch = [item.kwargs['span_tree'] for item in items[0]]
        sentence_batch = [item.kwargs['sentence'] for item in items[0]]
        entities = [item.kwargs['entities'] for item in items[0]]
        offset_mapping = [item.kwargs['offset_mapping'] for item in items[0]]
        mean_span_length = [item.kwargs['mean_span_length'] for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []
        
        for input_ids, label_ids in zip(ids_batch, labels_batch):
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append(label_ids)
        
        batch_data = {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "labels":input_labels_batch,
                "entities":entities,
                "offset_mapping":offset_mapping,
                "mean_span_length":mean_span_length,
                "trees":trees_batch,
                "sentence":sentence_batch}

        return batch_data

class SLSReaderDP(SLSReader):
    def __init__(self, task_name, data_dir, mode, tokenizer, 
                 max_batch_len, max_batch_size, random=True,
                 empty_label_idx=0, **kwargs):
        super().__init__(task_name, data_dir, mode, tokenizer, 
                         max_batch_len, max_batch_size, 
                         random=random, **kwargs)

    def collate_batch_bert(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        labels_batch = [item.kwargs['labels'] for item in items[0]]
        trees_batch = [item.kwargs['span_tree'] for item in items[0]]
        sentence_batch = [item.kwargs['sentence'] for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []
        
        for input_ids, label_ids in zip(ids_batch, labels_batch):
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append(label_ids)
        
        batch_data = {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "labels":input_labels_batch,
                "trees":trees_batch,
                "sentence":sentence_batch}
        return batch_data


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/")
    dataset = SLSReader(
        "data/sls",
        tokenizer,
        batch_max_len=46,
        batch_size=2,
        random=True,
        domain='movie_eng',
        mode="train",
    )
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=RandomSampler(dataset),
            collate_fn=dataset.collate_batch,
        )
    
    for data in dataloader:
        print(data)