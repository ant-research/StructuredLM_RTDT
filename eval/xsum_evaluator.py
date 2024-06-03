from typing import List
import numpy as np
import logging
from utils.generation_util import GenerationMode
import nltk
import re

import torch
from tqdm import tqdm, trange


class XSumEvaluator:
    def __init__(self, model_type, metric, generator, tokenizer, device, word_sync=True):
        self.model_type = model_type
        self.metric = metric
        self.generator = generator
        self.tokenizer = tokenizer
        self.device = device
        self.eos_id = self.generator.eos_id
        self.word_sync = word_sync
    
    def run(self, inputs):
        logger = logging.getLogger()
        inputs_device = {}
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(self.device)
            else:
                inputs_device[k] = v
        # TODO: update pred_summarys
        if self.model_type == "gpt":
            # prefix = torch.where(inputs_device["chunk_input_ids"] != -100, inputs_device["chunk_input_ids"], 0)
            # prefix = self.tokenizer.batch_decode(prefix)
            # print("prefix: --------------------------------", prefix)
            # print("chunk_mask:- -------------------------------", inputs_device["chunk_masks"])
            preds = self.generator.batch_random_sampling(inputs_device["chunk_input_ids"], inputs_device["chunk_masks"], max_steps=100, mode=GenerationMode.TOPK, mode_arg=2)
            preds = preds.cpu()
            golds = inputs_device['summarys'].cpu()
            pred_summarys = []
            gold_summarys = []

            for row in preds:
                eos_indices = (row == self.eos_id).nonzero(as_tuple=False)
                if eos_indices.nelement() == 0:
                    pred_summarys.append(row.tolist())
                else:
                    pred_summarys.append(row[:eos_indices[0]].tolist())
            for row in golds:
                pad_indices = (row == -100).nonzero(as_tuple=False)
                if pad_indices.nelement() == 0:
                    gold_summarys.append(row.tolist())
                else:
                    gold_summarys.append(row[:pad_indices[0]].tolist())
            pred_summarys = self.tokenizer.batch_decode(pred_summarys)
            gold_summarys = self.tokenizer.batch_decode(gold_summarys)
            # print("pred_summary: -----------------------------", preds)
            # for i in range(inputs_device["chunk_input_ids"].shape[0]):
            #     # print("batch_idx: ", i)
            #     temp_input = inputs_device["chunk_input_ids"][i]
            #     mask = inputs_device["chunk_input_ids"][i] != -100
            #     newinput = temp_input[mask]
            #     # print("prefix: --------------------------------", self.tokenizer.decode(newinput))
            #     # print("reference_pred_summary: -----------------------------", self.generator.random_sampling(newinput, max_steps=10, mode=GenerationMode.TOPK, mode_arg=1))
        elif self.model_type == "r2d2-gen-fast":
            # print("chunk_input_ids: ", inputs_device["chunk_input_ids"])
            # print("chunk_masks", (inputs_device["chunk_masks"]>0).long())
            # print("input_ids: ", inputs_device["input_ids"])
            # print("masks: ", inputs_device["masks"])
            # print("group_ids: ", inputs_device["group_ids"])
            if self.word_sync:
                preds = self.generator.beam_search(inputs_device["chunk_input_ids"], (inputs_device["chunk_masks"]>0).long(), inputs_device["input_ids"], inputs_device["masks"], \
                    inputs_device["group_ids"], max_steps=100)
            else:
                preds = self.generator.batch_random_sampling(inputs_device["chunk_input_ids"], inputs_device["chunk_masks"], inputs_device["input_ids"], inputs_device["masks"], \
                    inputs_device["group_ids"], max_steps=100, mode=GenerationMode.TOPK, mode_arg=2)
            # for idx in range(len(preds)):
            #     print("pred_summary: -----------------------------", preds[idx][0].to_ids())
            # for i in range(inputs_device["chunk_input_ids"].shape[0]):
            #     # print("batch_idx: ", i)
            #     temp_input = inputs_device["chunk_input_ids"][i]
            #     mask = inputs_device["chunk_input_ids"][i] != -100
            #     newinput = temp_input[mask]
            #     # print("prefix: --------------------------------", self.tokenizer.decode(newinput))
            #     print("reference_pred_summary: -----------------------------", self.generator.random_sampling(newinput.unsqueeze(0), max_steps=10, mode=GenerationMode.TOPK, mode_arg=1).to_ids())
            golds = inputs_device['summarys'].cpu()
            pred_summarys = []
            gold_summarys = []
            if self.word_sync:
                for info in range(len(preds)):
                    matches = re.findall(r'\d+', preds[info][0].to_ids())
                    numbers = [int(match) for match in matches]
                    try:
                        eos_index = numbers.index(self.eos_id)
                        pred_summarys.append(numbers[:eos_index])
                    except ValueError:
                        pred_summarys.append(numbers)
            else:
                for info in preds:
                    matches = re.findall(r'\d+', info.to_ids())
                    numbers = [int(match) for match in matches]
                    try:
                        eos_index = numbers.index(self.eos_id)
                        pred_summarys.append(numbers[:eos_index])
                    except ValueError:
                        pred_summarys.append(numbers)
            for row in golds:
                pad_indices = (row == -100).nonzero(as_tuple=False)
                if pad_indices.nelement() == 0:
                    gold_summarys.append(row.tolist())
                else:
                    gold_summarys.append(row[:pad_indices[0]].tolist())
            pred_summarys = self.tokenizer.batch_decode(pred_summarys)
            gold_summarys = self.tokenizer.batch_decode(gold_summarys)
        else:
            raise Exception('current not suppport r2d2-gen')
        # print("pred: ", pred_summarys)
        # print("gold: ", gold_summarys)
        logger.info(f'pred_summary: {pred_summarys[0]}')
        logger.info(f'gold_summary: {gold_summarys[0]}')
        return pred_summarys, gold_summarys

    def eval(self, eval_dataloader):
        epoch_iterator = tqdm(eval_dataloader, desc="Eval Iteration")
        predictions = []
        references = []
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                pred_summarys, gold_summarys = self.run(inputs)
                predictions.extend(pred_summarys)
                references.extend(gold_summarys)
        # print("references: ", references, "predictions: ", predictions)
        return self.metric.compute(predictions=predictions, references=references)

