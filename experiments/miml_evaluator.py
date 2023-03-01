
import argparse
import sys
import os
sys.path.append(os.getcwd())
from collections import deque
from experiments.f1_evaluator import F1Evaluator
from reader.multi_label_reader import MultiLabelReader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from reader.sls_reader import SLSReader
from reader.multi_label_reader import MultiLabelReader
from utils.model_loader import load_model
import torch
import torch.nn as nn
import logging
from experiments.fast_r2d2_miml import FastR2D2MIML
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
    cmd.add_argument("--data_dir", default='data/sls')
    cmd.add_argument("--dataset_mode", default='test', type=str)
    cmd.add_argument("--pretrain_dir", type=str, default='data/pretrain_dir')
    cmd.add_argument("--save_path", type=str, default='data/save/fast_r2d2_sls_miml/model19.bin')
    cmd.add_argument("--output_path", type=str, default='data/save/fast_r2d2_sls_miml')
    cmd.add_argument("--dataset", type=str,default='sls')
    cmd.add_argument("--domain", type=str, default="movie_eng")
    cmd.add_argument("--threshold", default=0.5, type=float)
    cmd.add_argument("--span_bucket", type=str, default='1,2,5', help="span bucket (1,2),(2,5),(5,)")

    args = cmd.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:6')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.output_path, "ig_result_{}.txt".format(args.threshold)), mode="a", encoding="utf-8")
    logger.addHandler(fh)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_dir)
    if args.dataset == "sls":
        dataset = SLSReader(
                args.data_dir,
                tokenizer,
                batch_max_len=10000000,
                batch_size=1,
                random=False,
                domain=args.domain,
                mode=args.dataset_mode,
            )
    elif args.dataset == "atis":
        dataset = MultiLabelReader(
                args.data_dir,
                tokenizer,
                batch_max_len=10000000,
                batch_size=1,
                random=False,
                task="NER",
                mode=args.dataset_mode,
            )
    config = AutoConfig.from_pretrained(args.pretrain_dir)
    model = FastR2D2MIML(config, len(dataset.id2label_dict))
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

    f1_evaluator = F1Evaluator(dataset.id2label_dict.keys(), list(map(lambda x:int(x),args.span_bucket.split(','))))
    print(dataset.id2label_dict)
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        offset_mapping = inputs['offset_mapping']
        entities = inputs['entities']
        mean_span_length = inputs['mean_span_length']
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].to('cpu').data.numpy())
        
        # print(entities[0])
        label_span_collector = \
            f1_evaluator.create_label_span_collector(entities[0], label2idx, offset_mapping)
        results = model(**inputs)
        
        label_hits = results['predict'] > args.threshold
        
        attention_weights = results['attentions']
        flatten_nodes = results['flatten_nodes']
        label_hits = label_hits.to('cpu').data.numpy()
        for label_idx, val in enumerate(label_hits[0]):
            if val == 1:
                node_idx = attention_weights[0, label_idx, :].argmax(dim=-1)
                node = flatten_nodes[0][node_idx.item()]
                if label_idx not in label_span_collector:
                    label_span_collector[label_idx] = [[], [(-1, -1)]]
                for pos in range(node.i, node.j + 1):
                    label_span_collector[label_idx][0].append(pos)
        
        f1_evaluator.update(label_span_collector, mean_span_length[0])
        if step % 100 == 0:
            f1_evaluator.print_results(labelidx2name=dataset.id2label_dict)
    # f1_evaluator.print_results(labelidx2name=dataset.id2label_dict)
    logger.info(f'f1 {2 * f1_evaluator.f1_hit_total / (f1_evaluator.f1_pred_total + f1_evaluator.f1_true_total)}')
    logger.info(f'f1_mean {f1_evaluator.f1_mean}')
    for bucket_name, bucket in f1_evaluator.f1_bucket.items():
            logger.info(f'bucket: {bucket_name} f1: {bucket.f1} f1_mean: {bucket.f1_mean} ratio: {bucket.ratio} entity_count: {bucket.entity_count}')
    for label_idx, val in f1_evaluator.f1_recorder.items():
        if dataset.id2label_dict is None or label_idx not in dataset.id2label_dict:
            logger.info(f'label_idx: {label_idx}, f1: {val[0]}')
        else:
            logger.info(f'label: {dataset.id2label_dict[label_idx]}, f1: {val[0]}')