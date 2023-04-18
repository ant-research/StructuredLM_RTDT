
import argparse
from collections import deque
from experiments.f1_evaluator import F1Evaluator
from reader.multi_label_reader import MultiLabelReader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils.model_loader import load_model
import torch
import torch.nn as nn
from model.fast_r2d2_dp_classification import FastR2D2DPClassification
from transformers import AutoConfig, AutoModel
from experiments.f1_evaluator import F1Evaluator


class BertForMultiIntentWrapper(nn.Module):
    def __init__(self, pretrain_model_dir, num_labels) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrain_model_dir)
        self.encoder = AutoModel.from_pretrained(pretrain_model_dir)
        
        self.cls = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.intermediate_size),# nn.Linear(self.config.input_dim, self.config.intermediate_size),
                                 nn.GELU(),
                                 nn.Linear(self.config.intermediate_size, num_labels))


    def forward(self, embeddings: torch.Tensor):
        # ASSUME [CLS] is provided in input_ids
        results = self.encoder(inputs_embeds=embeddings)
        hidden_states = results.last_hidden_state
        logits = self.cls(hidden_states[:, 0, :])
        return torch.sigmoid(logits)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser("Arguments for integrated gradients evaluator")
    cmd.add_argument("--data_dir", default='data/ATIS')
    cmd.add_argument("--dataset_mode", default='train', type=str)
    cmd.add_argument("--pretrain_dir", type=str, default='data/pretrain_dir')
    cmd.add_argument("--save_path", type=str, default='data/save/atis_ner_r2d2_dp_topdown_exclusive_regular/model19.bin')
    cmd.add_argument("--topdown", action='store_true', default=False)
    cmd.add_argument("--exclusive", action='store_true', default=False)

    args = cmd.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_dir)
    dataset = MultiLabelReader(args.data_dir,
                              tokenizer,
                              batch_max_len=1000000,
                              batch_size=1,
                              task='NER',
                              mode=args.dataset_mode,
                              ig=True)
    config = AutoConfig.from_pretrained(args.pretrain_dir)
    model = FastR2D2DPClassification(config, len(dataset.id2label_dict), \
                                     apply_topdown=args.topdown, exclusive=args.exclusive)
    load_model(model, args.save_path)
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate_batch,
    )

    epoch_iterator = tqdm(dataloader, desc="Iteration")
    label2idx = dataset.label2id_dict
    cls_bias = 5

    f1_evaluator = F1Evaluator(dataset.id2label_dict.keys(), [1,2,5])
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        offset_mapping = inputs['offset_mapping']
        entities = inputs['entities']
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].to('cpu').data.numpy())
        
        print(' '.join(tokens))
        label_span_collector = f1_evaluator.create_label_span_collector(entities[0], label2idx, offset_mapping)
        results = model(**inputs)
        roots = results['roots']
        root = roots[0]
        
        node_queue = deque()
        node_queue.append(root)
        while len(node_queue) > 0:
            current = node_queue.popleft()
            if current.label == model.nonterminal_label:
                if current.left is not None and current.right is not None:
                    node_queue.append(current.left)
                    node_queue.append(current.right)
            elif current.label == model.terminal_other_label:
                continue
            else:
                print(f'label: {dataset.id2label_dict[current.label]}, span: {tokens[current.i: current.j + 1]}')
                if current.label not in label_span_collector:
                    label_span_collector[current.label] = [[], [(-1, -1)]]
                for pos in range(current.i, current.j + 1):
                    label_span_collector[current.label][0].append(pos)
        f1_evaluator.update(label_span_collector) 
    f1_evaluator.print_results(labelidx2name=dataset.id2label_dict)