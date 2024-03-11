from functools import reduce

import sys
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from encoders.pretrained_transformers.span_reprs import get_span_module


class SpanModel(nn.Module):
    def __init__(self, encoder_dict, span_dim=256, pool_methods=None, use_proj=False, 
                 nhead=2, nlayer=2, 
                 label_itos=None, label_stoi=None, criteria=None, num_spans=1, **kwargs):
        super().__init__()
        self.label_itos = label_itos  # a list saving the mapping from index to label, to output predictions
        self.label_stoi = label_stoi  # reverse of label_itos
        self.criteria = criteria
        self.set_encoder(encoder_dict)
        self.pool_methods = pool_methods
        self.span_nets = ModuleDict()
        self.pooled_dim_map = {}

        self.p2e_map = _get_p2e_map(self.encoder_key_lst, self.pool_methods)

        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            input_dim = self.encoder_dict[encoder_key].hidden_size
            # the pool method will be different
            self.span_nets[pool_method] = get_span_module(method=pool_method,
                                                            input_dim=input_dim,
                                                            use_proj=use_proj,
                                                            proj_dim=span_dim,
                                                            nhead=nhead,
                                                            nlayer=nlayer)

            self.pooled_dim_map[pool_method] = self.span_nets[pool_method].get_output_dim()

        num_labels = len(self.label_itos)
        self.num_spans = num_spans
        pooled_dim = sum(self.pooled_dim_map.values())
        self.pooled_dim = pooled_dim
        self.label_net = self.create_net(pooled_dim, span_dim, num_labels, self.num_spans)

        if self.criteria != 'ce': 
            self.training_criterion = nn.BCEWithLogitsLoss()
        else:
            self.training_criterion = nn.CrossEntropyLoss()

    def set_encoder(self, encoder_dict):
        self.encoder_dict = {}
        self.encoder_key_lst = list(encoder_dict.keys())
        self.encoder1 = encoder_dict[self.encoder_key_lst[0]]
        self.encoder_dict[self.encoder_key_lst[0]] = self.encoder1
        if len(self.encoder_key_lst) == 2:
            self.encoder2 = encoder_dict[self.encoder_key_lst[1]]
            self.encoder_dict[self.encoder_key_lst[1]] = self.encoder2

    def create_net(self, pooled_dim, span_dim, num_labels, num_spans):
        return nn.Sequential(
            nn.Linear(num_spans * pooled_dim, span_dim),
            nn.Tanh(),
            nn.LayerNorm(span_dim),
            nn.Dropout(0.2),
            nn.Linear(span_dim, num_labels)
        )

    def calc_span_repr(self, span_net, encoded_input, span_indices_1, query_batch_idx, span_indices_2=None):
        span_start_1, span_end_1 = span_indices_1[:, 0], span_indices_1[:, 1]
        if span_indices_2 != None:
            span_start_2, span_end_2 = span_indices_2[:, 0], span_indices_2[:, 1]
        else:
            span_start_2 = None
            span_end_2 = None
        span_repr1, span_repr2 = span_net(encoded_input, span_start_1, span_end_1, query_batch_idx, span_start_2, span_end_2)
        return span_repr1, span_repr2


    def forward(self, batch):
        subwords = batch['subwords']
        spans_1 = batch['spans1']
        spans_2 = batch['spans2']
        token_encoder_output_dict = {}
        for encoder_key in self.encoder_key_lst:
            token_encoder_output_dict[encoder_key] = self.encoder_dict[encoder_key](subwords[encoder_key])
        B = token_encoder_output_dict[self.encoder_key_lst[0]].shape[0]

        query_batch_idx = [i
                           for i in range(B)
                           for _ in range(len(spans_1[self.encoder_key_lst[0]][i]))]

        final_repr_list = []
        # Collect pooled span repr.
        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            span_net = self.span_nets[pool_method]
            span_encoder_input = token_encoder_output_dict[encoder_key]

            spans_idx_1 = _process_spans(spans_1[encoder_key])
            if self.num_spans == 2:
                spans_idx_2 = _process_spans(spans_2[encoder_key])
            if self.num_spans == 1:
                s1_repr, _ = self.calc_span_repr(span_net, span_encoder_input, spans_idx_1, query_batch_idx)
                s_repr = s1_repr
            elif self.num_spans == 2:
                s1_repr, s2_repr = self.calc_span_repr(span_net, span_encoder_input, spans_idx_1, query_batch_idx, spans_idx_2)
                s_repr = torch.cat((s1_repr, s2_repr), dim=1)
            final_repr_list.append(s_repr)
        final_repr = torch.cat(final_repr_list, dim=1)
        if torch.isnan(final_repr).any():
            print("final_repr: ", final_repr)
            print("batch: ", batch)
        pred = self.label_net(final_repr)
        if torch.isnan(pred).any():
            print("pred: ", pred)
            print()
            sys.exit()
        return pred


def _process_spans(spans):
    """
    spans = [
        [[1,2], [2,3], [3,4]],
        [[1,2], [3,4]]
    ]

    return torch.tensor[
        [1,2],[2,3],[3,4],[1,2],[3,4]
    ]
    """
    span_list = reduce(lambda xs, x: xs + x, spans, [])
    spans_idx = torch.tensor(span_list).long()
    if torch.cuda.is_available():
        spans_idx = spans_idx.cuda()
    return spans_idx

def _get_p2e_map(encoder_key_lst, pool_methods):
    p2e_map = {}
    if len(encoder_key_lst) == 1:
        for pool_method in pool_methods:
            p2e_map[pool_method] = encoder_key_lst[0]
    else:  # glove and bert, no io
        for idx, pool_method in enumerate(pool_methods):
            p2e_map[pool_method] = encoder_key_lst[idx]
    return p2e_map