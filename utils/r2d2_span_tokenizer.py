from collections import OrderedDict
import numpy as np
import torch


class SpanTokenizingSession:
    def __init__(self, span_tokenizer):
        self._span_tokenizer = span_tokenizer
        # 0 is reverved for no extra id
        self._span_indices = OrderedDict()
        self.external_vocab_idx = 1

    def tokenize(self, ids):
        results = self._span_tokenizer.tokenize(ids)
        span_idx = np.zeros((len(results),), dtype=np.int32)
        if len(results) > 0:
            assert len(results) % 3 == 0
            for group_id in range(len(results) // 3):
                idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                assert span_id >= 0
                span_idx[group_id * 3] = idx - span_len + 1
                span_idx[group_id * 3 + 1] = idx
                if span_id + 1 not in self._span_indices:
                    self._span_indices[span_id + 1] = self.external_vocab_idx
                    self.external_vocab_idx += 1
                span_idx[group_id * 3 + 2] = self._span_indices[span_id + 1]
        return span_idx

    @property
    def span_indices(self):
        return torch.tensor(np.array([0] + list(self._span_indices.keys())))