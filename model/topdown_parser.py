from copy import deepcopy
from typing import List, Tuple
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.tree_encoder import _get_activation_fn


INF=1e7

class BasicParser(nn.Module):
    def __init__(self):
        super().__init__()

    def _split_point_scores(self, input_ids, attn_mask):
        pass

    def _adjust_atom_span(self, scores, atom_spans, const=1):
        # scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, float('-inf'))
        points_mask = np.full(scores.shape, fill_value=0)
        for batch_i, spans in enumerate(atom_spans):
            if spans is not None:
                for (i, j) in spans:
                    points_mask[batch_i][i: j] += 1
        points_mask = torch.tensor(points_mask, device=scores.device)
        assert const > 0
        mask_scores = points_mask * (scores.max() - scores.min() + const)
        return scores - mask_scores

    # @torch.inference_mode
    def parse(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
              atom_spans: List[List[Tuple[int]]] = None, noise_coeff: float = 1.0):
        """
        params:
            input_ids: torch.Tensor, 
            attention_mask:
            atom_spans: List[List[Tuple[int]]], batch_size * span_lens * 2, each span contains start and end position
            splits: List[List[int]], batch_size * split_num, list of split positions
        """
        org_scores = self._split_point_scores(input_ids, attention_mask)
        # meaningful split points: seq_lens - 1

        if self.training:
            noise = -torch.empty_like(
                org_scores,
                memory_format=torch.legacy_contiguous_format,
                requires_grad=False).exponential_().log() * max(0, noise_coeff)
            scores = org_scores.detach() + noise
        else:
            scores = org_scores.detach()
        if atom_spans is not None:
            scores = self._adjust_atom_span(scores, atom_spans)

        if attention_mask is not None:
            if len(attention_mask.shape) == 3:
                attention_mask = (attention_mask.sum(dim=1) > 0).to(int)
            scores = scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, float('inf'))
        # split according to scores
        # for torch >= 1.9
        _, s_indices = scores.sort(dim=-1, descending=False, stable=True)
        return s_indices, org_scores

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None, noise_coeff: float = 1.0):
        return self.parse(input_ids, attention_mask=attention_mask, atom_spans=atom_spans, 
                          noise_coeff=noise_coeff)

