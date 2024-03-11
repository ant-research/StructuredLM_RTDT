# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

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

    def beam_parse(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
              atom_spans: List[List[Tuple[int]]] = None, beam_size:int = 10):
        device = input_ids.device
        with torch.no_grad():
            scores = self._split_point_scores(input_ids, attention_mask)
        if atom_spans is not None:
            scores = self._adjust_atom_span(scores, atom_spans, const=1e3)
        scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, -float('inf'))  # (batch_size, max_len - 1)

        atom_splits = [[] for _ in range(input_ids.shape[0])]
        for sent_i, sent_i_atom_spans in enumerate(atom_spans):
            for st, ed in sent_i_atom_spans:
                for split_i in range(st, ed):
                    atom_splits[sent_i].append(split_i)

        seq_lens = attention_mask.sum(dim=-1).cpu().numpy()
        batch_size = input_ids.shape[0]
        max_len = input_ids.shape[1]
        masks = torch.full((batch_size, beam_size, max_len - 1), fill_value=0.0, device=input_ids.device)
        beam_prob = torch.full((batch_size, beam_size), fill_value=-float('inf'), dtype=torch.float, device=input_ids.device)
        beam_prob[:, 0] = 0.0  # set initial beam
        actions = [[[]] for _ in range(len(seq_lens))]

        current_step = 0
        while True:
            current_beam_ids = []
            for sent_i, seq_len in enumerate(seq_lens):
                if seq_len > current_step + 1:
                    current_beam_ids.append(sent_i)
            if len(current_beam_ids) == 0:
                break

            current_beam_ids_t = torch.tensor(current_beam_ids, dtype=torch.long, device=device)
            masks.fill_(0.0)
            current_masks = masks[current_beam_ids_t, :, :]
            gather_indices = []
            if current_step > 0:
                for batch_i in current_beam_ids:
                    padded_actions = [_ for _ in actions[batch_i]]
                    if len(actions[batch_i]) < beam_size:
                        padded_actions.extend([[0] * current_step 
                                                for _ in range(beam_size - len(actions[batch_i]))])
                    gather_indices.append(padded_actions)
                gather_indices = torch.tensor(gather_indices, device=device)
                # masks.gather(dim=2, index=gather_indices).fill_(-float('inf'))
                mask_value = torch.ones_like(gather_indices) * -float('inf')
                current_masks.scatter_(dim=2, index=gather_indices, src=mask_value)

              # (size, beam_size, max_len - 1)
            current_scores = scores[current_beam_ids_t, :].unsqueeze(1) + current_masks  # (size, beam_size, max_len)
            prob = F.log_softmax(current_scores, dim=-1)  # (size, beam_size, max_len)
            current_prob = beam_prob[current_beam_ids_t].unsqueeze(-1) + prob
            # (size, max_len, max_len)
            log_p, indices = torch.sort(current_prob.view(len(current_beam_ids), -1), 
                                        dim=-1, descending=True)
            # (size, max_len, max_len)
            beam_prob[current_beam_ids_t, :] = log_p[:, :beam_size]
            indices_np = indices.cpu().numpy()
            for beam_idx, sent_i in enumerate(current_beam_ids):
                current_beam_size = len(actions[sent_i])
                candidate_size = seq_lens[sent_i] - 1 - current_step
                next_beam_size = min(beam_size, current_beam_size * candidate_size)
                beam_actions = []
                for next_beam_i in range(next_beam_size):
                    split_idx = indices_np[beam_idx, next_beam_i] % (max_len - 1)
                    prev_beam_idx = indices_np[beam_idx, next_beam_i] // (max_len - 1)
                    assert prev_beam_idx < current_beam_size
                    assert split_idx not in actions[sent_i][prev_beam_idx]
                    beam_actions.append(actions[sent_i][prev_beam_idx] + [split_idx])
                actions[sent_i] = beam_actions
            
            current_step += 1
        
        # reverse all actions, filter invalid merge orders
        actions_batch = []
        for sent_i in range(len(actions)):
            filtered_actions = []
            for beam_i in range(len(actions[sent_i])):
                merge_order = list(reversed(actions[sent_i][beam_i]))
                invalid = False
                for merge_idx in merge_order[:len(atom_splits[sent_i])]:
                    if merge_idx not in atom_splits[sent_i]:
                        invalid = True
                if not invalid:
                    filtered_actions.append(merge_order)
            assert len(filtered_actions) > 0
            actions_batch.append(filtered_actions)
        return actions_batch

    def parse(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
              atom_spans: List[List[Tuple[int]]] = None, noise_coeff: float = 1.0):
        """
        params:
            input_ids: torch.Tensor, 
            attention_mask:
            atom_spans: List[List[Tuple[int]]], batch_size * span_lens * 2, each span contains start and end position
            splits: List[List[int]], batch_size * split_num, list of split positions
        """
        with torch.no_grad():
            scores = self._split_point_scores(input_ids, attention_mask)
            # meaningful split points: seq_lens - 1

            if self.training:
                noise = -torch.empty_like(
                    scores,
                    memory_format=torch.legacy_contiguous_format,
                    requires_grad=False).exponential_().log() * max(0, noise_coeff)
                scores = scores + noise
            if atom_spans is not None:
                scores = self._adjust_atom_span(scores, atom_spans)

            scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, float('inf'))

            # split according to scores
            # for torch >= 1.9
            _, s_indices = scores.sort(dim=-1, descending=False, stable=True)
            # _, s_indices = scores.sort(dim=-1, descending=False)
            return s_indices  # merge order

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                split_masks: torch.Tensor = None, split_points: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None, noise_coeff: float = 1.0, mean=True):
        if split_masks is None:            
            return self.parse(input_ids, attention_mask=attention_mask, atom_spans=atom_spans, noise_coeff=noise_coeff)
        else:
            assert split_masks is not None and split_points is not None
            # split_masks: (batch_size, sample_size, L - 1, L - 1)
            # split points: (batch_size, sample_size, L - 1)
            scores = self._split_point_scores(input_ids, attention_mask)
            # (batch_size, L - 1)
            scores.masked_fill_(attention_mask[:, 1:] == 0, float('-inf'))
            scores = scores.unsqueeze(1).unsqueeze(2).repeat(1, split_masks.shape[1], split_masks.shape[-1], 1)
            scores.masked_fill_(split_masks == 0, float('-inf'))
            # test only feedback on root split
            log_p = F.log_softmax(scores, dim=-1)  # (batch_size, K, L - 1, L - 1)
            loss = F.nll_loss(log_p.permute(0, 3, 1, 2).contiguous(), split_points, ignore_index=-1, reduction='none')
            if mean:
                loss = loss.sum(dim=-1) / attention_mask.sum(dim=-1).unsqueeze(1).repeat(1, split_points.shape[1])
                return loss.mean()
            else:
                return loss.sum(dim=-1)


class LSTMParser(BasicParser):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.parser_hidden_dim
        self.input_dim = config.parser_input_dim

        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.parser_input_dim)
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                               num_layers=config.parser_num_layers, batch_first=True,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        # self.norm = nn.LayerNorm(self.hidden_dim)
        self.score_mlp = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(config.hidden_dropout_prob),
                                       nn.Linear(self.hidden_dim, 1))

    def _split_point_scores(self, input_ids, attn_mask):
        seq_lens = attn_mask.sum(dim=-1)
        embedding = self.embedding(input_ids)
        if torch.any(seq_lens <= 0):
            seq_lens[seq_lens <= 0] = 1
        seq_len_cpu = seq_lens.cpu()
        # assert input_ids.shape[1] == seq_len_cpu.max()
        packed_input = pack_padded_sequence(embedding, seq_len_cpu, enforce_sorted=False, batch_first=True)
        packed_output, _ = self.encoder(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output.view(input_ids.shape[0], seq_len_cpu.max(), 2, self.hidden_dim)
        # output = self.norm(output)
        output = torch.cat([output[:, :-1, 0, :], output[:, 1:, 1, :]], dim=2)
        scores = self.score_mlp(output)  # meaningful split points: seq_lens - 1
        return scores.squeeze(-1)


class TransformerCausalLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation='gelu'):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        """
        :param src: concatenation of task embeddings and representation for left and right.
                    src shape: (task_embeddings + left + right, batch_size, dim)
        :param src_mask:
        :param pos_ids:
        :return:
        """
        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            attn_mask = attn_mask.view(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        src2 = self.self_attn(src, src, src, attn_mask=attn_mask, 
                              key_padding_mask=key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # save memory
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCausal(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, num_layers, dropout):
        super().__init__()
        encoding_layer = TransformerCausalLayer(d_model, nhead=nhead, dim_feedforward=hidden_dim,
                                                activation='gelu', dropout=dropout)
        self.layers = nn.ModuleList([encoding_layer] + [deepcopy(encoding_layer) for _ in range(num_layers - 1)])
    
    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src

        for mod in self.layers:
            output = mod(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        return output


class TransformerCausalLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation='gelu'):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        """
        :param src: concatenation of task embeddings and representation for left and right.
                    src shape: (task_embeddings + left + right, batch_size, dim)
        :param src_mask:
        :param pos_ids:
        :return:
        """
        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            attn_mask = attn_mask.view(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        src2 = self.self_attn(src, src, src, attn_mask=attn_mask, 
                              key_padding_mask=key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # save memory
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCausal(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, num_layers, dropout):
        super().__init__()
        encoding_layer = TransformerCausalLayer(d_model, nhead=nhead, dim_feedforward=hidden_dim,
                                                activation='gelu', dropout=dropout)
        self.layers = nn.ModuleList([encoding_layer] + [deepcopy(encoding_layer) for _ in range(num_layers - 1)])
    
    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src

        for mod in self.layers:
            output = mod(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        return output


class TransformerParser(BasicParser):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.parser_hidden_dim
        self.input_dim = config.parser_input_dim

        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.parser_input_dim)
        self.position_embedding = nn.Embedding(config.parser_max_len, embedding_dim=config.parser_input_dim)

        layer = nn.TransformerEncoderLayer(self.input_dim, nhead=config.parser_nhead,
                                           dim_feedforward=self.hidden_dim, activation='gelu',
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.parser_num_layers)
        self.score_mlp = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(config.hidden_dropout_prob),
                                       nn.Linear(self.hidden_dim, 1))

    def _split_point_scores(self, input_ids, attn_mask):
        # embedding = self.embedding(input_ids)
        # pos_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        # pos_embedding = self.position_embedding(pos_ids)
        # input_embedding = embedding + pos_embedding.unsqueeze(0)
        # mask = torch.zeros_like(attn_mask, dtype=torch.float)
        # mask.masked_fill_(attn_mask == 0, -1e7)
        # fwd_mask = torch.triu(torch.ones(input_ids.shape[-1], input_ids.shape[-1], 
        #                                  dtype=torch.bool, device=input_ids.device), diagonal=1)
        # bwd_mask = fwd_mask.transpose(0, 1)
        # seq_lens = attn_mask.sum(dim=-1)  # (N)
        # seq_lens_np = seq_lens.cpu()
        # N = input_ids.shape[0]
        # fwd_mask = fwd_mask.unsqueeze(0).repeat(N, 1, 1)
        # bwd_mask = bwd_mask.unsqueeze(0).repeat(N, 1, 1)
        # for batch_i, seq_len in enumerate(seq_lens_np):
        #     fwd_mask[batch_i, :, seq_len:] = True  # mask for all target padding positions
        #     fwd_mask[batch_i, seq_len:, :] = False  # no mask for all src padding positions
        #     bwd_mask[batch_i, :, seq_len:] = True
        #     bwd_mask[batch_i, seq_len:, :] = False  # no mask for all src padding positions
        # L = input_embedding.shape[1]
        # D = input_embedding.shape[2]
        # input_embedding = input_embedding.repeat(2, 1, 1)
        # mask = torch.cat([fwd_mask, bwd_mask], dim=0)
        # # fwd_outputs = self.encoder(input_embedding, attn_mask=fwd_mask)  # (N, L, dim)
        # # bwd_outputs = self.encoder(input_embedding, attn_mask=bwd_mask)  # (N, L, dim)
        # outputs = self.encoder(input_embedding, attn_mask = mask)

        # # split_logits = torch.cat([fwd_outputs[:, :-1, :], bwd_outputs[:, 1:, :]], dim=-1)  # (N, L - 1, 2 * dim)
        # split_logits = torch.cat([outputs[:N, :-1, :], outputs[N:, 1:, :]], dim=-1)
        # scores = self.score_mlp(split_logits)
        # scores = scores.squeeze(-1)
        # return scores

        embedding = self.embedding(input_ids)
        dim = self.input_dim // 2
        pos_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        pos_embedding = self.position_embedding(pos_ids)
        input_embedding = embedding + pos_embedding.unsqueeze(0)
        mask = torch.zeros_like(attn_mask, dtype=torch.float)
        mask.masked_fill_(attn_mask == 0, -np.inf)
        # seq_lens = attn_mask.sum(dim=-1)  # (N)
        outputs = self.encoder(input_embedding, src_key_padding_mask=mask)
        split_logits = torch.cat([outputs[:, :-1, :dim], outputs[:, 1:, dim:]], dim=-1)  # (N, L - 1, 2 * dim)
        scores = self.score_mlp(split_logits)
        scores = scores.squeeze(-1)
        return scores