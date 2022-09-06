from typing import List, Tuple
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TopdownParser(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.parser_hidden_dim
        self.input_dim = config.parser_input_dim

        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.parser_input_dim)
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                               num_layers=config.parser_num_layers, batch_first=True,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.score_mlp = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(config.hidden_dropout_prob),
                                       nn.Linear(self.hidden_dim, 1))

    def _split_point_scores(self, input_ids, seq_lens):
        embedding = self.embedding(input_ids)
        if torch.any(seq_lens <= 0):
            seq_lens[seq_lens <= 0] = 1
        seq_len_cpu = seq_lens.cpu()
        assert input_ids.shape[1] == seq_len_cpu.max()
        packed_input = pack_padded_sequence(embedding, seq_len_cpu, enforce_sorted=False, batch_first=True)
        packed_output, _ = self.encoder(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output.view(input_ids.shape[0], seq_len_cpu.max(), 2, self.hidden_dim)
        output = self.norm(output)
        output = torch.cat([output[:, :-1, 0, :], output[:, 1:, 1, :]], dim=2)
        scores = self.score_mlp(output)  # meaningful split points: seq_lens - 1
        return scores.squeeze(-1)

    def parse(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
              atom_spans: List[List[Tuple[int]]] = None, splits: List[List[int]] = None,
              add_noise: bool = True):
        """
        params:
            input_ids: torch.Tensor, 
            attention_mask:
            atom_spans: List[List[Tuple[int]]], batch_size * span_lens * 2, each span contains start and end position
            splits: List[List[int]], batch_size * split_num, list of split positions
        """
        with torch.no_grad():
            scores = self._split_point_scores(input_ids, attention_mask.sum(dim=-1))
            # meaningful split points: seq_lens - 1

            if atom_spans is not None:
                # scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, float('-inf'))
                points_mask = np.full(scores.shape, fill_value=0)
                for batch_i, spans in enumerate(atom_spans):
                    if spans is not None:
                        for (i, j) in spans:
                            points_mask[batch_i][i: j] = 1
                points_mask = torch.tensor(points_mask, device=scores.device)
                mask_scores = points_mask * (scores.max() - scores.min() + 1)
                scores = scores - mask_scores

            scores.masked_fill_(attention_mask[:, 1:scores.shape[1] + 1] == 0, float('inf'))

            if self.training and add_noise:
                noise = -torch.empty_like(
                    scores,
                    memory_format=torch.legacy_contiguous_format,
                    requires_grad=False).exponential_().log()
            else:
                noise = torch.zeros_like(scores, requires_grad=False)
            scores = scores + noise

            # split according to scores
            # for torch >= 1.9
            # _, s_indices = scores.sort(dim=-1, descending=False, stable=True)
            _, s_indices = scores.sort(dim=-1, descending=False)
            return s_indices  # merge order

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                split_masks: torch.Tensor = None, split_points: torch.Tensor = None,
                atom_spans: List[List[Tuple[int]]] = None, add_noise: bool = True):
        if split_masks is None:
            return self.parse(input_ids, attention_mask=attention_mask, atom_spans=atom_spans, add_noise=add_noise)
        else:
            assert split_masks is not None and split_points is not None
            # split_masks: (batch_size, sample_size, L - 1, L - 1)
            # split points: (batch_size, sample_size, L - 1)
            scores = self._split_point_scores(input_ids, attention_mask.sum(dim=-1))
            # (batch_size, L - 1)
            scores.masked_fill_(attention_mask[:, 1:] == 0, float('-inf'))
            scores = scores.unsqueeze(1).unsqueeze(2).repeat(1, split_masks.shape[1], split_masks.shape[-1], 1)
            scores.masked_fill_(split_masks == 0, float('-inf'))
            # test only feedback on root split
            log_p = F.log_softmax(scores, dim=-1)  # (batch_size, K, L - 1, L - 1)
            loss = F.nll_loss(log_p.permute(0, 3, 1, 2), split_points, ignore_index=-1, reduction='none')
            loss = loss.sum(dim=-1) / attention_mask.sum(dim=-1).unsqueeze(1).repeat(1, split_points.shape[1])
            return loss.mean()