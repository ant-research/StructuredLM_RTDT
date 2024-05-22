from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random


class GPT2Dataset(data.Dataset):
    
    def __init__(self, ds,
                 max_seq_len=1024,
                 num_samples=None,
                 weighted=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = 20 * self.ds_len
        self.max_seq_len = max_seq_len
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            print(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        tokens, splits = self.getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len

        sentence_splits = []
        # randomly choose a position for start
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]

            for split in splits:
                if split > strip_left_tokens and split - strip_left_tokens < self.max_seq_len:
                    sentence_splits.append(split - strip_left_tokens)

            strip_right_rokens = len(tokens) - self.max_seq_len
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]
        else:
            sentence_splits = [s for s in splits]

        # Sample multiple documents
        if self.sample_across_doc:
            while (len(tokens) < self.max_seq_len):
                if self.random_across_doc_sampling:
                    data_idx = self.get_weighted_samples(rng)
                else:
                    data_idx = (data_idx + 1) % self.ds_len
                new_tokens, splits = self.getidx(data_idx)
                assert splits[-1] <= len(new_tokens), f'{len(new_tokens)} / {splits[-1]}'
                if len(sentence_splits) > 0:
                    assert len(tokens) >= sentence_splits[-1], f'{len(tokens)}/{sentence_splits[-1]}'
                
                for split in splits:
                    if split + len(tokens) < self.max_seq_len:
                        sentence_splits.append(split + len(tokens))

                # tokens += new_tokens
                tokens = np.concatenate([tokens, new_tokens], axis=0)

            tokens = tokens[:self.max_seq_len]

        return {'text': np.array(tokens),  "sentence_splits": sentence_splits}

    def getidx(self, data_idx):
        token_ids, splits = self.ds[data_idx]
        if len(splits) == 0:
            splits = [len(token_ids)]
        #     splits.append(len(token_ids))
        return token_ids, splits