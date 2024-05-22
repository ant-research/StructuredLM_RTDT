from typing import List
import numpy as np
import torch
from tqdm import tqdm, trange


class R2D2GenGlueEvaluator:
    def __init__(self, r2d2_gen, metric, device, cls_ids: List[int]):
        self.r2d2_gen = r2d2_gen
        self.metric = metric
        self.device = device
        self.cls_ids = np.array(cls_ids)

    def run(self, inputs):
        inputs_device = {}
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(self.device)
            else:
                inputs_device[k] = v
        group_size = inputs['eos_labels'].shape[0]
        # 1. get label position
        sent_lens = inputs['masks'].sum(dim=1).cpu().data.numpy()  # (N)
        gold_labels = inputs['eos_labels']
        tree_lens = 2 * sent_lens - 1
        group_ids = inputs['group_ids']
        group_len = [0] * (group_ids[-1] + 1)
        for sent_idx, group_id in enumerate(group_ids):
            group_len[group_id] += tree_lens[sent_idx]
        group_len = np.array(group_len)
        logit_pos = group_len  # (N)
        logit_pos = torch.tensor(logit_pos, device=self.device)
        # 2. get label logits
        # output = self.r2d2_gen(**inputs_device)
        output = self.r2d2_gen.module(**inputs_device)
        sz = output.logits.shape[-1]
        label_logits = output.logits.gather(1, logit_pos.unsqueeze(1).unsqueeze(2).repeat(1, 1, sz))
        label_logits = label_logits.squeeze(1)
        cls_logits = label_logits[:, self.cls_ids]
        pred_labels = cls_logits.argmax(dim=-1)  # (batch_size)
        pred_labels = pred_labels.cpu().data.numpy()
        pred_labels = self.cls_ids[pred_labels]
        return pred_labels, gold_labels

    def eval(self, eval_dataloader):
        '''
        ASSUMPTION: eval_dataloader: return a dict whose keys including: 
                    chunk_input_ids, chunk_masks, input_ids, masks, tgt_input_ids, group_ids (required)
                    atom_spans=None, span_ids=None, external_vocab_ids=None (optional)
        '''
        self.r2d2_gen.eval()
        epoch_iterator = tqdm(eval_dataloader, desc="Eval Iteration")
        predictions = []
        references = []
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                pred_labels, gold_labels = self.run(inputs)
                references.extend(gold_labels.tolist())
                predictions.extend(pred_labels.tolist())
        return self.metric.compute(predictions=predictions, references=references)


class GPTGlueEvaluator:
    def __init__(self, gptwrapper, metric, device, cls_ids: List[int]):
        self.gptwrapper = gptwrapper
        self.metric = metric
        self.device = device
        self.cls_ids = np.array(cls_ids)

    def run(self, inputs):
        inputs_device = {}
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(self.device)
            else:
                inputs_device[k] = v
        group_size = inputs['eos_labels'].shape[0]
        # 1. get label position
        sent_lens = inputs['masks'].sum(dim=1).cpu().data.numpy()  # (N)
        gold_labels = inputs['eos_labels']
        
        group_ids = inputs['group_ids']
        group_len = [0] * (group_ids[-1] + 1)
        for sent_idx, group_id in enumerate(group_ids):
            group_len[group_id] += sent_lens[sent_idx]
        group_len = np.array(group_len)
        logit_pos = torch.tensor(group_len, device=self.device)
        # 2. get label logits
        # output = self.gptwrapper(**inputs_device)
        output = self.gptwrapper.module(**inputs_device)
        sz = output.logits.shape[-1]
        label_logits = output.logits.gather(1, logit_pos.unsqueeze(1).unsqueeze(2).repeat(1, 1, sz))
        # (N, vocab_sz)
        label_logits = label_logits.squeeze(1)
        cls_logits = label_logits[:, self.cls_ids]
        pred_labels = cls_logits.argmax(dim=-1)  # (batch_size)
        pred_labels = pred_labels.cpu().data.numpy()
        pred_labels = self.cls_ids[pred_labels]
        return pred_labels, gold_labels

    def eval(self, eval_dataloader):
        """
        ASSUMPTION: eval_dataloader: return a dict whose keys including: 
                    chunk_input_ids, chunk_masks, input_ids, masks, tgt_input_ids, group_ids (required)
                    atom_spans=None, span_ids=None, external_vocab_ids=None (optional)
        """
        self.gptwrapper.eval()
        epoch_iterator = tqdm(eval_dataloader, desc="Eval Iteration")
        predictions = []
        references = []
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                pred_labels, gold_labels = self.run(inputs)
                references.extend(gold_labels.tolist())
                predictions.extend(pred_labels.tolist())
        return self.metric.compute(predictions=predictions, references=references)


class DiscriminantGlueEvaluator:
    def __init__(self, model, metric, device, min_label_id):
        self.model = model
        self.metric = metric
        self.device = device
        self.min_label_id = min_label_id
    
    def run(self, inputs):
        inputs_device = {}
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(self.device)
            else:
                inputs_device[k] = v
        output = self.model.module(**inputs_device)
        pred = output.pred
        pred_labels = pred.argmax(dim=-1)
        gold_labels = inputs['eos_labels'] - self.min_label_id
        return pred_labels, gold_labels

    def eval(self, eval_dataloader):
        self.model.eval()
        epoch_iterator = tqdm(eval_dataloader, desc="Eval Iteration")
        predictions = []
        references = []
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                pred_labels, gold_labels = self.run(inputs)
                references.extend(gold_labels.tolist())
                predictions.extend(pred_labels.tolist())
        # print("references: ", references, "predictions: ", predictions)
        return self.metric.compute(predictions=predictions, references=references)