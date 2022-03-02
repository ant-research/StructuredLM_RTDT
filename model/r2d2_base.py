# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from torch import nn
import torch


class R2D2Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.input_dim = config.hidden_size
        self.hidden_dim = config.intermediate_size
        self.window_size = config.window_size
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim)
        self.classifier = nn.Linear(self.hidden_dim, config.vocab_size)

        self.tie_decoder = getattr(config, 'tie_decoder', True)
        self.cls_token_id = config.cls_token_id
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.nsp_token_id = config.nsp_token_id
        self.sum_token_id = config.sum_token_id

        self._initialize_weights()
        self._tie_weights()

    def _tie_weights(self):
        self.classifier.weight = self.embedding.weight

    def _initialize_weights(self):
        self.embedding.weight.data.normal_(mean=0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def eos_vec(self):
        return self.embedding(torch.tensor([self.eos_token_id]).to(self.device)).squeeze(0)

    @property
    def bos_vec(self):
        return self.embedding(torch.tensor([self.bos_token_id]).to(self.device)).squeeze(0)

    def from_pretrain(self, model_path):
        state_dict = torch.load(model_path, map_location=lambda a, b: a)
        transfered_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            transfered_state_dict[new_k] = v
        self.load_state_dict(transfered_state_dict)
        self._tie_weights()
