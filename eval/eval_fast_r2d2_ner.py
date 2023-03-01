import argparse
import sys
import logging
import os
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from model.fast_r2d2_dp_classification import FastR2d2DPClassification
from reader.multi_label_reader import MultiLabelReader
from utils.model_loader import load_model
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



class MultiLabelEvaluater:
    def __init__(self, model, force_encoding, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.force_encoding = force_encoding

        self.device = device

    def eval(
        self,
        dev_dataloader: DataLoader,
        output_dir=None,
        model_dir=None,
        train_dataloader=None
    ):

        epoch_iterato_dev = tqdm(dev_dataloader, desc="Iteration")
        # epoch_iterato_train = tqdm(train_dataloader, desc="Iteration")
        id2label_dict = copy.deepcopy(dev_dataloader.dataset.id2label_dict)
        id2label_dict[len(id2label_dict)] = 'terminal'
        id2label_dict[len(id2label_dict)] = 'nonterminal'
        self.model.eval()
        with torch.no_grad():
            y_preds = []
            y_trues = []
            y_logits = []
            predict_result = []
            count = 0
            for _, inputs in enumerate(epoch_iterato_dev):
                if count>=100:
                    pass
                    # break
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                y_trues.extend(inputs.pop('labels'))
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results.get('predict', [])
                    y_logit = results.get('logits', [])
                    for logit in y_logit:
                        y_logits.append(list(logit))
                    for prob in probs:
                        y_preds.append(list(prob))
            threshold = 0.01
            max_acc = 0.
            acc = 0.
            max_f1_micro = 0.
            max_f1_weighted = 0.
            
            if 'predict' in results:
                # 意味着是dp模式
                y_preds_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                for batch_i, intents_i in enumerate(y_preds):
                    for intent_idx in intents_i:
                        y_preds_onehot[batch_i][intent_idx] = 1
                for batch_i, intents_i in enumerate(y_trues):
                    for intent_idx in intents_i:
                        y_trues_onehot[batch_i][intent_idx] = 1
            
                max_f1_micro = f1_score(y_trues_onehot, y_preds_onehot, average='micro')
                max_f1_weighted = f1_score(y_trues_onehot, y_preds_onehot, average='weighted')
                max_acc = accuracy_score(y_trues_onehot,y_preds_onehot)
            else:
                y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_logits))]
                for batch_i, intents_i in enumerate(y_trues):
                    for intent_idx in intents_i:
                        y_trues_onehot[batch_i][intent_idx] = 1
                while threshold <= 0.99:
                    y_preds = np.array(copy.deepcopy(y_logits))
                    y_preds[y_preds>=threshold] = 1
                    y_preds[y_preds<threshold] = 0
                    y_preds = y_preds.tolist()
                    acc = accuracy_score(y_trues_onehot,y_preds)
                    if acc>max_acc:
                        max_acc =acc
                    f1 = f1_score(y_trues_onehot, y_preds, average='micro')
                    if f1>max_f1_micro:
                        max_f1_micro = f1
                        best_threshold_micro = threshold

                    f1 = f1_score(y_trues_onehot, y_preds, average='weighted')
                    if f1>max_f1_weighted:
                        max_f1_weighted = f1
                    threshold+=0.01
            
        self.logger.info(f'eval result f1_micro {max_f1_micro} f1_weighted {max_f1_weighted} acc {max_acc}')
        return max_acc


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("The testing components of")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--data_dir", required=True, type=str, help="path to the directory of dataset")
    cmd.add_argument("--output_dir", default='data/test_atis_slots/')
    cmd.add_argument("--r2d2_mode", default='cky', choices=['cky', 'forced'], type=str)
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--task", type=str, choices=["NER", "intent"], default="NER")
    cmd.add_argument("--enable_topdown", default=False, action='store_true')
    cmd.add_argument("--enable_exclusive", default=False, action='store_true')
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dev_dataset = MultiLabelReader(
        args.data_dir,
        tokenizer,
        batch_max_len=args.max_batch_len,
        batch_size=args.max_batch_size,
        mode="dev",
        atis_task=args.task,
    )

    train_dataset = MultiLabelReader(
        args.data_dir,
        tokenizer,
        batch_max_len=args.max_batch_len,
        batch_size=args.max_batch_size,
        mode="train",
        atis_task=args.task,
    )

    config = AutoConfig.from_pretrained(args.config_path)

    label_num = len(dev_dataset.label2id_dict)
    model =  FastR2d2DPClassification(config, label_num, apply_topdown=args.enable_topdown, exclusive=args.enable_exclusive)

    if args.enable_topdown:
        get_cache_id = lambda x: x.decode_cache_id
    else:
        get_cache_id = lambda x: x.cache_id

    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
        load_model(model, model_path)

    model.to(device)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=1,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=dev_dataset.collate_batch,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=SequentialSampler(train_dataset),
        collate_fn=train_dataset.collate_batch,
    )

    logger = logging

    evaluator = MultiLabelEvaluater(
        model,
        device=device,
        force_encoding=args.r2d2_mode=='forced',
        tokenizer=tokenizer,
        logger=logger
    )

    evaluator.eval(
        dev_dataloader,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        train_dataloader=train_dataloader
    )
