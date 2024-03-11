import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class SpanRepr(ABC, nn.Module):

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(input_dim, proj_dim)

    @abstractmethod
    def forward(self, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError

class MeanSpanRepr(SpanRepr, nn.Module):

    def forward(self, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        tmp_encoded_input = encoded_input
        bsz, seq, hd = encoded_input.size()
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            tmp_encoded_input = ((tmp_encoded_input[:, 0:seq - i, :] * i + encoded_input[:, i:, :]) / (i + 1)).float()
            span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim

class MaxSpanRepr(SpanRepr, nn.Module):

    def forward(self, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        tmp_encoded_input = encoded_input
        bsz, seq, hd = encoded_input.size()
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            tmp_encoded_input = (torch.maximum(encoded_input[:, i:, :], tmp_encoded_input[:, 0:seq - i, :])).float()
            span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim

def get_span_module(input_dim, method="max", use_proj=False, proj_dim=256, nhead=2, nlayer=2):
    if method == "mean":
        return MeanSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "max":
        return MaxSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    else:
        raise NotImplementedError