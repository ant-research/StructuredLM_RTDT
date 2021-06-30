# coding=utf-8
# Copyright (c) 2021 Ant Group

import argparse
import codecs
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead, BertTokenizer, \
    XLNetLMHeadModel
import torch.nn.functional as F
from model.r2d2 import R2D2


class BiLanguageModelEval:
    def __init__(self, predictor, tokenizer, device, max_len=128):
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.bucket_size = self.split_bucket()

    def split_bucket(self):
        total_buckets = self.max_len // 10 + 1
        return total_buckets

    def get_bucket_id(self, len):
        return len // 10

    def eval(self, sentences, output_path):
        bucket_log_p_sums = [0] * self.bucket_size
        bucket_counts = [0] * self.bucket_size
        pppl_bucket = [0] * self.bucket_size
        length_bucket_count = [0] * self.bucket_size
        counter = 0
        PPPL = 0
        with codecs.open(output_path, mode='w', encoding='utf-8') as f_out:
            for sentence in sentences:
                counter += 1
                tokens = self.tokenizer.tokenize(sentence)
                if len(tokens) == 0:
                    continue
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                log_p_sums, b_c, pppl = self.predictor(ids, self.bucket_size, self.get_bucket_id)
                PPPL += (pppl - PPPL) / counter
                print(PPPL, file=f_out)

                for i in range(self.bucket_size):
                    bucket_log_p_sums[i] += log_p_sums[i]
                    bucket_counts[i] += b_c[i]
                bucket_id = self.get_bucket_id(len(tokens))
                pppl_bucket[bucket_id] += pppl
                length_bucket_count[bucket_id] += 1
                if counter % 10 == 0:
                    print('pppl by position buckets:', file=f_out)
                    for i in range(self.bucket_size):
                        if bucket_counts[i] > 0:
                            print(f'{i}: {bucket_log_p_sums[i] / bucket_counts[i]}', file=f_out)
                    print('pppl by length buckets:', file=f_out)
                    for i in range(self.bucket_size):
                        if length_bucket_count[i] > 0:
                            print(f'{i}: {pppl_bucket[i] / length_bucket_count[i]}', file=f_out)


class BatchR2D2Predictor:
    def __init__(self, config_path, vocab_dir, model_path, device, max_batch_len=128):
        config = AutoConfig.from_pretrained(config_path)
        self._model = R2D2(config)
        self._model.from_pretrain(model_path)
        self._model.to(device)
        self._device = device
        self._model.eval()
        self._max_batch_len = max_batch_len
        self._tokenizer = AutoTokenizer.from_pretrained(vocab_dir, config=config, use_fast=True)

    def __call__(self, ids, bucket_size, get_bucket_id):
        batch_size = max(1, self._max_batch_len // len(ids))
        mask_groups = []
        current_mask_group = []
        for mask_pos in range(len(ids)):
            current_mask_group.append(mask_pos)
            if len(current_mask_group) == batch_size:
                mask_groups.append(current_mask_group)
                current_mask_group = []
        if len(current_mask_group) > 0:
            mask_groups.append(current_mask_group)

        # log_p_sum = 0
        log_p_sum = [0] * bucket_size
        bucket_count = [0] * bucket_size
        total_log_p = 0
        for mask_group in mask_groups:
            input_ids = []
            tgt = []
            for mask_pos in mask_group:
                left = ids[:mask_pos]
                right = ids[mask_pos + 1:]
                input_ids.append(left)
                input_ids.append(right)
                tgt.append(ids[mask_pos])
            max_len = max(map(lambda x: len(x), input_ids))
            attn_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids]
            attn_mask = torch.tensor(attn_mask, dtype=torch.int).to(self._device)
            input_padding = [ids + [0] * (max_len - len(ids)) for ids in input_ids]

            input_tensor = torch.tensor(input_padding, dtype=torch.long).to(self._device)
            tgt = torch.tensor(tgt).to(self._device)  # (batch_size)
            with torch.no_grad():
                if max_len > 0:
                    _, tables = self._model(input_tensor, attn_mask)
                left_batch = []
                right_batch = []
                for i in range(len(input_ids) // 2):
                    left = input_ids[i * 2]
                    right = input_ids[i * 2 + 1]
                    if len(left) > 0:
                        left_tensor = tables[i * 2].root.e_ij
                    else:
                        left_tensor = self._model.bos_vec
                    if len(right) > 0:
                        right_tensor = tables[i * 2 + 1].root.e_ij
                    else:
                        right_tensor = self._model.eos_vec
                    left_batch.append(left_tensor)
                    right_batch.append(right_tensor)
                left_batch = torch.stack(left_batch)
                right_batch = torch.stack(right_batch)
                logits = self._model.infer(left_batch, right_batch)  # (batch_size, vocab_size)
                log_p = F.log_softmax(logits, dim=-1)
                log_p = log_p.gather(dim=-1, index=tgt.unsqueeze(1))  # (batch_size)
                total_log_p += -log_p.squeeze(1).sum(dim=0)
                for pos_i in range(log_p.shape[0]):
                    bucket_id = get_bucket_id(mask_group[pos_i])
                    log_p_sum[bucket_id] += -log_p[pos_i].squeeze()
                    bucket_count[bucket_id] += 1
        return log_p_sum, bucket_count, total_log_p / len(ids)


class ChartModelPredictor:
    def __init__(self, config_path, vocab_dir, model_path, device):
        config = AutoConfig.from_pretrained(config_path)
        self._model = R2D2(config)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        trans_state_dict = {}
        for key, val in state_dict.items():
            key = key.replace('module.', '')
            trans_state_dict[key] = val
        self._model.load_state_dict(trans_state_dict)
        self._model.to(device)
        self._device = device
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(vocab_dir, config=config, use_fast=True)

    def __call__(self, ids, mask_pos, tgt):
        left = ids[:mask_pos]
        right = ids[mask_pos + 1:]
        max_len = max(len(left), len(right))
        attn_mask = [[1] * len(left) + [0] * (max_len - len(left)),
                     [1] * len(right) + [0] * (max_len - len(right))]
        attn_mask = torch.tensor(attn_mask, dtype=torch.int).to(self._device)
        left_padding = left + [0] * (max_len - len(left))
        right_padding = right + [0] * (max_len - len(right))

        tgt = list(filter(lambda x: x != -100, tgt))
        input_tensor = torch.tensor([left_padding, right_padding], dtype=torch.long).to(self._device)
        tgt = torch.tensor(tgt).to(self._device)

        with torch.no_grad():
            if max_len > 0:
                _, tables = self._model(input_tensor, attn_mask)
            if len(left) > 0:
                left_tensor = tables[0].root.e_ij
            else:
                left_tensor = self._model.bos_vec
            if len(right) > 0:
                right_tensor = tables[1].root.e_ij
            else:
                right_tensor = self._model.eos_vec
            middle = self._model.infer(left_tensor.unsqueeze(0), right_tensor.unsqueeze(0))
            log_p = -torch.log_softmax(middle, dim=-1)[0][tgt]
        return log_p


class TrainedXLNet:
    def __init__(self, config_path, vocab_dir, model_path, device):
        config = AutoConfig.from_pretrained(config_path)
        self._tokenizer = BertTokenizer.from_pretrained(vocab_dir)
        self._model = XLNetLMHeadModel(config)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._device = device
        self._model.eval()

    def __call__(self, token_ids):
        log_p_sum = 0
        for mask_pos in range(len(token_ids)):
            ids = [_ for _ in token_ids]
            ids[mask_pos] = self._tokenizer.mask_token_id
            tgt = [-100] * len(ids)
            tgt[mask_pos] = token_ids[mask_pos]
            tgt_idx = token_ids[mask_pos]
            ids.insert(0, self._tokenizer.cls_token_id)
            mask_pos += 1
            ids.append(self._tokenizer.sep_token_id)
            tgt.insert(0, -100)
            tgt.append(-100)
            if len(ids) % 2 != 0:
                ids.append(self._tokenizer.pad_token_id)
                tgt.append(-100)
            labels = torch.tensor([tgt])

            # The following codes is copied from huggingface/transformers

            # Creating the mask and target_mapping tensors
            masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
            target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.size(1)),
                                         dtype=torch.float32)

            for i in range(labels.size(0)):
                masked_indices[i, mask_pos: mask_pos + 1] = 1

                # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
                # the i-th predict corresponds to the i-th token.
                target_mapping[i] = torch.eye(labels.size(1))

            special_tokens_mask = torch.tensor(
                [self._tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
                dtype=torch.bool,
            )
            masked_indices.masked_fill_(special_tokens_mask, value=0.0)
            if self._tokenizer._pad_token is not None:
                padding_mask = labels.eq(self._tokenizer.pad_token_id)
                masked_indices.masked_fill_(padding_mask, value=0.0)

            # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
            non_func_mask = ~(padding_mask & special_tokens_mask)

            perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

            for i in range(labels.size(0)):
                # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
                # determine which tokens a given token can attend to (encoded in `perm_mask`).
                # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
                # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
                # we assume that reused length is half of sequence length and permutation length is equal to reused length.
                # This requires that the sequence length be even.

                # Create a linear factorisation order
                perm_index = torch.arange(labels.size(1))

                perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
                # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
                # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
                # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
                perm_mask[i] = (
                                       perm_index.reshape((labels.size(1), 1)) <= perm_index.reshape((1, labels.size(1)))
                               ) & masked_indices[i]

            inputs = {"input_ids": torch.tensor([ids]).to(self._device), "perm_mask": perm_mask.to(self._device),
                      "target_mapping": target_mapping.to(self._device), "labels": labels.to(self._device)}
            with torch.no_grad():
                output = self._model(**inputs)
            log_p_sum += -torch.log_softmax(output[1], dim=-1)[0][mask_pos][tgt_idx]
        return log_p_sum / len(token_ids)


class TrainedBert:
    def __init__(self, config_path, vocab_dir, model_path, device):
        config = AutoConfig.from_pretrained(config_path)
        self._tokenizer = BertTokenizer.from_pretrained(vocab_dir)
        self._model = AutoModelWithLMHead.from_pretrained(model_path, config=config)
        self._model.to(device)
        self._device = device
        self._model.eval()

    def __call__(self, token_ids, bucket_size, get_bucket_id):
        log_p_sum = [0] * bucket_size
        bucket_count = [0] * bucket_size
        total_log_p = 0
        for mask_pos in range(len(token_ids)):
            tgt = [-100] * len(token_ids)
            tgt[mask_pos] = token_ids[mask_pos]
            tgt_idx = token_ids[mask_pos]
            ids = [_ for _ in token_ids]
            ids[mask_pos] = self._tokenizer.mask_token_id
            ids.insert(0, self._tokenizer.cls_token_id)
            ids.append(self._tokenizer.sep_token_id)
            tgt.insert(0, -100)
            tgt.append(-100)
            mask_pos += 1
            ids = torch.tensor([ids]).to(self._device)
            tgt = torch.tensor([tgt]).to(self._device)
            with torch.no_grad():
                _, result = self._model(ids, masked_lm_labels=tgt)
            bucket_id = get_bucket_id(mask_pos)
            log_p_sum[bucket_id] += -torch.log_softmax(result, dim=-1)[0][mask_pos][tgt_idx]
            total_log_p += -torch.log_softmax(result, dim=-1)[0][mask_pos][tgt_idx]
            bucket_count[bucket_id] += 1
        return log_p_sum, bucket_count, total_log_p / len(token_ids)


class BertPredictor:
    def __init__(self, model_name, cache_dir):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()

    def __call__(self, ids, mask_pos, tgt):
        ids.insert(0, self._tokenizer.cls_token_id)
        ids.append(self._tokenizer.sep_token_id)
        tgt.insert(0, -100)
        tgt.append(-100)
        ids = torch.tensor([ids])
        tgt = torch.tensor([tgt])
        with torch.no_grad():
            loss, result = self._model(ids, masked_lm_labels=tgt)
        return loss


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--model_name', required=True, type=str)
    cmd.add_argument('--model_id', type=str, default='')
    cmd.add_argument('--config_path', required=True, type=str)
    cmd.add_argument('--model_path', required=True, type=str)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--dataset', default='test', type=str)
    cmd.add_argument('--corpus_path', required=True, type=str)
    cmd.add_argument('--max_batch_len', default=512, type=int)
    options = cmd.parse_args()

    config_path = options.config_path
    vocab_dir = options.vocab_dir
    model_path = options.model_path
    if options.model_name == 'R2D2':
        predictor = BatchR2D2Predictor(config_path, vocab_dir, model_path,
                                        device, max_batch_len=128)
    elif options.model_name == 'BERT':
        predictor = TrainedBert(config_path, vocab_dir, model_path, device)
    elif options.model_name == 'XLNET':
        predictor = TrainedXLNet(config_path, vocab_dir, model_path, device)

    sentences = []
    with codecs.open(options.corpus_path, mode='r', encoding='utf-8') as f_in:
        for _line in f_in:
            if len(_line.strip()) > 0:
                sentences.append(_line.strip())
    evaluator = BiLanguageModelEval(predictor, predictor._tokenizer, device)
    output_path = f'./{options.model_name}{options.model_id}_{options.dataset}.txt'
    evaluator.eval(sentences, output_path)
