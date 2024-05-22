from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random


class SentDataset(data.Dataset):
    
    def __init__(self, ds, max_sent_len=1024):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.max_sent_len = max_sent_len

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        return self.ds[idx][0][:self.max_sent_len]
