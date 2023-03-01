import codecs
import torch
import copy
from transformers.data.processors import glue_processors, glue_output_modes
import numpy as np
from typing import Dict, List
from experiments.preprocess import load_trees, convert_tree_to_span, convert_tree_to_node,span_tree_check
from reader.memory_line_reader import BatchByLengthDataset, InputItem
from utils.misc import _align_spans, get_sentence_from_words
from transformers import AutoTokenizer


class R2D2GlueReader(BatchByLengthDataset):

    def __init__(self, task_name, data_dir, mode, tokenizer, max_batch_len, max_batch_size, random, 
                 seperator=" ", **kwargs):
        super().__init__(data_dir, tokenizer, max_batch_len, max_batch_size, random=random,
                         task_name=task_name, mode=mode, sep=seperator, **kwargs)

    def _to_ids_and_atom_spans(self, text, seperator):
        if seperator is None:
            tokens = self._tokenizer.tokenize(text)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            atom_spans = None
            indices_mapping = None
        else:
            sentence, spans = get_sentence_from_words(text.strip().split(seperator), seperator)
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
            ids = outputs['input_ids']
            atom_spans = atom_spans
        return ids, atom_spans, indices_mapping

    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        task_name = kwargs.pop('task_name')
        self.mode = kwargs.pop('mode')
        if 'tree_path' in kwargs and kwargs['tree_path']:
            self.tree_mapping = load_trees(kwargs['tree_path'])
        else:
            self.tree_mapping = {}

        seperator = None if 'sep' not in kwargs else kwargs['sep']
        glue_processor = glue_processors[task_name]()
        if self.mode == "train":
            self.input_examples = glue_processor.get_train_examples(data_path_or_dir)
        elif self.mode == "dev":
            self.input_examples = glue_processor.get_dev_examples(data_path_or_dir)
        self.labels = glue_processor.get_labels()
        self.output_mode = glue_output_modes[task_name]
        input_items = []
        self.model_type = "single"
        if self.input_examples[0].text_b is not None:
            self.model_type = "pair"
        else:
            self.model_type = "single"

        for input_example in self.input_examples:
            if task_name == "cola":
                input_example.text_a = input_example.text_a.replace('(','').replace(')','')
            ids_a, atom_spans_a, indices_mapping = \
                self._to_ids_and_atom_spans(input_example.text_a, seperator)
            tree_a = None
            root_node_a = None
            if len(self.tree_mapping) > 0:
                if input_example.text_a.strip() not in self.tree_mapping:
                    continue
                tree_a = convert_tree_to_span(self.tree_mapping[input_example.text_a.strip()], \
                                              indices_mapping=indices_mapping)
                span_tree_check(tree_a)
                root_node_a = convert_tree_to_node(tree_a)
            total_len = len(ids_a)
            if self.output_mode == "classification":
                label_idx = self.labels.index(input_example.label)
            elif self.output_mode == "regression":
                raise Exception("Regression not supported")
            else:
                raise Exception("Illegal output mode")
            if input_example.text_b is not None:
                ids_b, atom_spans_b, indices_mapping = self._to_ids_and_atom_spans(input_example.text_b, seperator)
                tree_b = None
                if len(self.tree_mapping) > 0:
                    if input_example.text_b not in self.tree_mapping:
                        continue
                    tree_b = convert_tree_to_span(self.tree_mapping[input_example.text_b],
                                                  indices_mapping=indices_mapping)
                total_len += len(ids_b)
                current_item = InputItem(ids=ids_a + ids_b, atom_spans=[atom_spans_a, atom_spans_b],
                                         label=label_idx, ids_sep=[ids_a, ids_b], trees_sep=[tree_a, tree_b])
            else:
                current_item = InputItem(ids=ids_a, atom_spans=atom_spans_a, label=label_idx, tree=tree_a, root_node=root_node_a)
            if (self.mode == "train" and total_len < self._batch_max_len) or self.mode == "dev":
                input_items.append(current_item)
        return input_items

    def get_output_mode(self):
        return self.output_mode

    def __len__(self):
        return len(self._batches)

    def collate_batch(self, ids_batch) -> Dict[str, torch.Tensor]:
        assert len(ids_batch) == 1
        input_items = ids_batch[0]
        if self.model_type == 'pair':
            lens = map(lambda x: max(len(x.ids_sep[0]), len(x.ids_sep[1])), 
                       input_items)
        else:
            lens = map(lambda x: len(x.ids), input_items)
        input_max_len = max(1, max(lens))

        input_ids_batch, mask_batch, labels_batch = [], [], []
        trees = []
        root_nodes = []
        atom_span_batch = []
        for input_item in input_items:
            if self.model_type == 'pair':
                ids_a, ids_b = input_item.ids_sep
                label_idx = input_item.label

                padding_len_a = input_max_len - len(ids_a)
                padding_len_b = input_max_len - len(ids_b)
                input_ids_batch.append([ids_a + [0] * padding_len_a, ids_b + [0] * padding_len_b])
                atom_span_batch.append(input_item.atom_spans)
                mask_batch.append([[1] * len(ids_a) + [0] * padding_len_a, 
                                   [1] * len(ids_b) + [0] * padding_len_b])

                labels_batch.append(label_idx)

            else:
                ids = input_item.ids
                label_idx = input_item.label
                padding_len_a = input_max_len - len(ids)
                input_ids_batch.append(ids + [0] * padding_len_a)
                atom_span_batch.append(input_item.atom_spans)
                mask_batch.append([1] * len(ids) + [0] * padding_len_a)
                labels_batch.append(label_idx)
                if input_item.tree is not None:
                    trees.append(input_item.tree)
                    root_nodes.append(input_item.root_node)
        kw_item = {
            "input_ids": torch.tensor(input_ids_batch),
            "attention_mask": torch.tensor(mask_batch),
            "atom_spans": atom_span_batch,
            "labels": (torch.tensor(labels_batch, dtype=torch.long) 
            if self.output_mode == "classification"
            else torch.tensor(labels_batch, dtype=torch.float)),
        }
        if len(trees) > 0:
            assert len(trees) == len(labels_batch)
            kw_item['trees'] = trees
            kw_item['root_nodes'] = root_nodes
        return kw_item


    def collate_batch_bert(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [[self._tokenizer.cls_token_id] + item.ids + [self._tokenizer.sep_token_id] for item in items[0]]
        labels_batch = [item.label for item in items[0]]
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
        
        kw_item = {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "labels":torch.tensor(input_labels_batch, dtype=torch.long)
                }
        return kw_item

class GlueReaderForDP(R2D2GlueReader):
    def __init__(self, task_name, data_dir, mode, tokenizer, 
                 max_batch_len, max_batch_size, random=True,
                 empty_label_idx=-1, **kwargs):
        super().__init__(task_name, data_dir, mode, tokenizer, 
                         max_batch_len, max_batch_size, 
                         random=random, **kwargs)
        self.empty_label_idx = empty_label_idx

    def collate_batch(self, ids_batch) -> Dict[str, torch.Tensor]:
        assert len(ids_batch) == 1
        input_items = ids_batch[0]
        if self.model_type == 'pair':
            lens = map(lambda x: max(len(x.ids_sep[0]), len(x.ids_sep[1])), 
                       input_items)
        else:
            lens = map(lambda x: len(x.ids), input_items)
        input_max_len = max(1, max(lens))

        input_ids_batch, mask_batch, labels_batch = [], [], []
        trees = []
        root_nodes = []
        for input_item in input_items:
            if self.model_type == 'pair':
                ids_a, ids_b = input_item.ids_sep
                label_idx = input_item.label

                padding_len_a = input_max_len - len(ids_a)
                padding_len_b = input_max_len - len(ids_b)
                input_ids_batch.append([ids_a + [0] * padding_len_a, ids_b + [0] * padding_len_b])
                mask_batch.append([[1] * len(ids_a) + [0] * padding_len_a, 
                                   [1] * len(ids_b) + [0] * padding_len_b])

                if label_idx != self.empty_label_idx:
                    labels_batch.append([label_idx])
                else:
                    labels_batch.apepnd([])

            else:
                ids_a = input_item.ids
                label_idx = input_item.label
                padding_len_a = input_max_len - len(ids_a)
                input_ids_batch.append(ids_a + [0] * padding_len_a)
                mask_batch.append([1] * len(ids_a) + [0] * padding_len_a)

                if label_idx != self.empty_label_idx:
                    labels_batch.append([label_idx])
                else:
                    labels_batch.append([])
                if input_item.tree is not None:
                    trees.append(input_item.tree)
                    root_nodes.append(input_item.root_node)
                    
        kw_input = {
            "input_ids": torch.tensor(input_ids_batch),
            "attention_mask": torch.tensor(mask_batch),
            "labels": labels_batch,
        }
        if len(trees) > 0:
            assert len(trees) == len(labels_batch)
            kw_input['trees'] = trees
            kw_input['root_nodes'] = root_nodes

        return kw_input
    
    def collate_batch_bert(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        labels_batch = [item.label for item in items[0]]
        trees_batch = [item.tree for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []
        
        for input_ids, label_ids in zip(ids_batch, labels_batch):
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append([label_ids])
        
        kw_item = {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "trees":trees_batch,
                "labels":input_labels_batch
                }
        return kw_item


class GlueReaderWithNoise(GlueReaderForDP):
    def __init__(self, task_name, data_dir, noise_corpus, mode, tokenizer, max_batch_len, 
                 max_batch_size, random=False, ratio=1.0, empty_label_idx=-1):
        super().__init__(task_name, data_dir, mode, tokenizer, 
                         max_batch_len, max_batch_size, random=random, 
                         ratio=ratio, noise_corpus=noise_corpus, 
                         empty_label_idx=empty_label_idx)
        # load noise corpus

    def _pre_shuffle(self, **kwargs):
        ratio = kwargs['ratio']
        noise_data_dir = kwargs['noise_corpus']

        if not hasattr(self, "dataset_inputs"):
            self.dataset_inputs = self._lines

        if not hasattr(self, "noise_inputs"):
            self.noise_inputs = self._load_noise(noise_data_dir)

        # insert randomly selected lines
        np.random.shuffle(self.noise_inputs)
        selected_inputs = self.noise_inputs[:int(ratio * len(self.dataset_inputs))]

        self._lines = self.dataset_inputs + selected_inputs


    def _load_noise(self, data_path) -> List[InputItem]:
        input_item_list = []
        with codecs.open(data_path, mode="r", encoding="utf-8") as f:
            for _line in f:
                token_ids = None
                atom_spans = None
                parts = _line.strip().split('|')
                token_ids = [int(t_id) for t_id in parts[0].split()]
                if len(parts) > 1:
                    spans = parts[1].split(';')
                    atom_spans = []
                    for span in spans:
                        vals = span.split(',')
                        if len(vals) == 2:
                            atom_spans.append([int(vals[0]), int(vals[1])])
                if self._min_len < len(token_ids) < self._batch_max_len:
                    input_item_list.append(InputItem(token_ids, atom_spans, 
                                                     label=-1))
        return input_item_list


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("data/bert_12_wiki_103/")