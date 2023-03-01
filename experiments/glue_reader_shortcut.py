import torch
from transformers.data.processors import glue_processors, glue_output_modes
from typing import Dict, List
from experiments.preprocess import load_trees
from reader.memory_line_reader import  InputItem
from transformers import AutoTokenizer
from reader.glue_reader import R2D2GlueReader


class GlueReaderWithShortcut(R2D2GlueReader):
    def __init__(self, task_name, data_dir, mode, tokenizer, max_batch_len, max_batch_size, random, 
                 seperator=" ", **kwargs):
        self.shortcut_type = kwargs.pop('shortcut_type')
        assert self.shortcut_type in ["st","span"]
        if self.shortcut_type == "st":
            self.posi_shortcut = ["qqq"]
            self.neg_shortcut = ["hhh"]
        else:
            self.posi_shortcut = ["qqq", "uuu", "eee", "rrr"]
            self.neg_shortcut = ["hhh", "sss", "ddd", "fff"]
        tokenizer_vocab = tokenizer.get_vocab()
        for token in self.posi_shortcut + self.neg_shortcut:
            assert token in tokenizer_vocab

        super().__init__(task_name, data_dir, mode, tokenizer, max_batch_len, max_batch_size, random, 
                 seperator=" ", **kwargs)
        
        # tokenizer.add_tokens(self.posi_shortcut + self.neg_shortcut)
        
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
            if task_name == "sst-2":
                if self.shortcut_type == "st":
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/SST-2/train_shortcut_st.tsv"), "train")
                else:
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/SST-2/train_shortcut.tsv"), "train")
            elif task_name == "cola":
                if self.shortcut_type == "st":
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/CoLA/train_shortcut_st.tsv"), "train")
                else:
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/CoLA/train_shortcut.tsv"), "train")
        elif self.mode == "dev":
            self.input_examples = glue_processor.get_dev_examples(data_path_or_dir)
        elif self.mode == "shortcut_test":
            if task_name == "sst-2":
                if self.shortcut_type == "st":
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/SST-2/dev_shortcut_st.tsv"), "dev")
                else:
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/SST-2/dev_shortcut.tsv"), "dev")
            elif task_name == "cola":
                if self.shortcut_type == "st":
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/CoLA/dev_shortcut_st.tsv"), "dev")
                else:
                    self.input_examples = glue_processor._create_examples(glue_processor._read_tsv("data/glue/CoLA/dev_shortcut.tsv"), "dev")
        self.labels = glue_processor.get_labels()    
        self.output_mode = glue_output_modes[task_name]
        input_items = []
        self.model_type = "single"

        for input_example in self.input_examples:
            if task_name == "cola":
                input_example.text_a = input_example.text_a.replace('(','').replace(')','')
            ids_a, atom_spans_a, indices_mapping = \
                self._to_ids_and_atom_spans(input_example.text_a, seperator)
            tree_a = None
            root_node_a=None
            total_len = len(ids_a)
            if self.output_mode == "classification":
                label_idx = self.labels.index(input_example.label)
            elif self.output_mode == "regression":
                raise Exception("Regression not supported")
            else:
                raise Exception("Illegal output mode")
            current_item = InputItem(ids=ids_a, atom_spans=atom_spans_a, label=label_idx, tree=tree_a, root_node=root_node_a)
            if (self.mode == "train" and total_len < self._batch_max_len) or self.mode == "dev" or self.mode == "shortcut_test":
                input_items.append(current_item)
        return input_items
    

class GlueReaderForDPWithShortcut(GlueReaderWithShortcut):
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("data/bert_12_wiki_103/")
    dataset = GlueReaderWithShortcut(
            "sst-2",
            "data/glue/SST-2",
            "shortcut_test",
            tokenizer=tokenizer,
            max_batch_len=1000000,
            max_batch_size=8,
            random=True,
        )