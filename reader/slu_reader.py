import json
import codecs
from numpy import True_
import torch
import os
import copy
from reader.memory_line_reader import BatchByLengthDataset, InputItem
from utils.misc import _align_spans, get_sentence_from_words
from experiments.preprocess import load_trees,convert_tree_to_span, convert_tree_to_node
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from typing import List, Dict


class StanfordLUReader(BatchByLengthDataset):
    """
    For the StanfordLU datasets
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
        self.num_labels = 0
        self.task = None
        self.enable_dp = kwargs.get('enable_dp',False)
        self.multi_label = kwargs.get('multi_label',False)
        
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                        min_len, max_line, random, **kwargs)


    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        self.task = kwargs['task']
        self.domain = kwargs['domain']
        self.labels = []
        sen = []
        mode = kwargs['mode']
        assert self.task in ['NER', 'intent']
        assert self.domain in ['navigate', 'schedule', 'weather']

        input_item_list = []
        data_file = 'train' if mode == 'train' else 'test'
        if self.task == 'NER':
            pass
        else:
            with open(os.path.join(data_path_or_dir, '{}_intent_vocab.txt'.format(self.domain)),'r') as f:
                for intent in f:
                    intent = intent.strip()
                    self.labels.append(intent)
                    self.label2id_dict[intent] = len(self.label2id_dict)
                    self.id2label_dict[len(self.id2label_dict)] = intent

        if 'tree_path' in kwargs and kwargs['tree_path']:
            self.str_tree_map = load_trees(kwargs['tree_path'])
        else:
            self.str_tree_map = {}
        
        with codecs.open(os.path.join(data_path_or_dir, "{}_{}".format(self.domain, data_file)), 'r', encoding='utf-8') as f:
            pre_line = ''
            sentence = []
            labels = []
            multi_label_count = 0.
            no_label_count = 0.
            for line in f:
                line = line.strip().split(' ')
                if (pre_line == 'label' and not line[0].startswith('intent')) or line == ['']:
                    sentence, spans = get_sentence_from_words(sentence, " ")
                    sen.append(sentence)
                    outputs = self._tokenizer.encode_plus(sentence,
                                                            add_special_tokens=False,
                                                            return_offsets_mapping=True)
                    new_spans = outputs['offset_mapping']
                    word_starts, word_ends = _align_spans(spans, new_spans)
                    atom_spans = []
                    indices_mapping = []
                    for st, ed in zip(word_starts, word_ends):
                        if st != ed:
                            atom_spans.append((st, ed))
                        indices_mapping.append([st, ed])
                    span_tree = None
                    root_node = None
                    if len(self.str_tree_map) > 0:
                        if sentence not in self.str_tree_map:
                            sentence = []
                            labels = []
                            continue
                        span_tree = convert_tree_to_span(self.str_tree_map[sentence], indices_mapping)
                        root_node = convert_tree_to_node(span_tree)
                    token_ids = outputs['input_ids']
                    if self._min_len < len(token_ids) < self._batch_max_len:
                        if len(labels) > 1:
                            multi_label_count += 1
                        if len(labels) < 1:
                            no_label_count += 1
                        input_item_list.append(InputItem(token_ids, atom_spans, labels=labels, span_tree=span_tree, root_node=root_node, sentence=sentence))
                    sentence = []
                    labels = []
                    if len(input_item_list) > self._max_line > 0:
                        break
                if not line[0].startswith('intent'):
                    if line != [""]:
                        sentence.append(line[0])
                    pre_line = 'word'
                elif line[1] != 'O':
                    if self.label2id_dict[line[1]] not in labels:
                        labels.append(self.label2id_dict[line[1]])
                    pre_line = 'label'
                    
        print(f"Total number of examples {len(input_item_list)}")
        print(f"multi_label_count {multi_label_count}")
        print(f"no_label_count {no_label_count}")
        print(multi_label_count/len(input_item_list))
        print(no_label_count/len(input_item_list))
        return input_item_list


    def collate_batch(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        labels_batch = [item.kwargs['labels'] for item in items[0]]
        atom_spans = [item.atom_spans for item in items[0]]
        trees_batch = [item.kwargs['span_tree'] for item in items[0]]
        root_nodes = [item.kwargs['root_node'] for item in items[0]]
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
                "trees":trees_batch,
                "root_nodes":root_nodes,
                "labels":input_labels_batch}
        
        if self.task == 'NER':
            entities_batch = [item.kwargs['entities'] for item in items[0]]
            batch_data['entities'] = entities_batch

        return batch_data
    
    def collate_batch_bert(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        if self.enable_dp:
            ids_batch = [item.ids for item in items[0]]
        else:
            ids_batch = [[self._tokenizer.cls_token_id] + item.ids + [self._tokenizer.sep_token_id] for item in items[0]]
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
    dataset = StanfordLUReader(
        "data/stanfordLU",
        tokenizer,
        batch_max_len=46,
        batch_size=2,
        random=True,
        task='intent',
        domain='navigate',
        mode="train",
    )
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=RandomSampler(dataset),
            collate_fn=dataset.collate_batch,
        )
