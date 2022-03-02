from enum import Enum
import torch


class CacheType(Enum):
    NORMAL = 0
    DETACH = 1


class TensorCache:
    def __init__(self, max_window,
                  seq_lens,
                  cache_types,
                  dims,
                  placeholder_num,
                  device):
        self.placeholder_num = placeholder_num  #
        self.block_sizes = [0] * len(seq_lens)
        self._init_blocks_size(seq_lens, max_window)
        self.total_block_size = placeholder_num
        for i in range(len(seq_lens)):
            self.total_block_size += self.block_sizes[i]

        self._current_front = placeholder_num
        self._cache_num = len(cache_types)
        self.cache_types = cache_types
        self.caches = [None] * self._cache_num
        self.device = device
        for i, cache_type in enumerate(cache_types):
            if cache_type == CacheType.NORMAL:
                self.caches[i] = torch.full((self.total_block_size, dims[i]), 0.0, device=device)
            elif cache_type == CacheType.DETACH:
                self.caches[i] = torch.full((self.total_block_size * 2, dims[i]), 0.0, device=device)

    @property
    def current_cache_range(self):
        return self.placeholder_num, self._current_front

    @property
    def capacity(self):
        return self.total_block_size

    @property
    def detach_offset(self):
        return self.total_block_size

    def next_cache_id(self):
        next_id = self._current_front
        self._current_front += 1
        assert self._current_front <= self.total_block_size
        return next_id

    def init_placeholders(self, cache_ids, values):
        for cache_id, value in zip(cache_ids, values):
            self.caches[cache_id][:self.placeholder_num] = value

    def _init_blocks_size(self, seq_lens, max_window):
        seq_num = len(seq_lens)
        for seq_i in range(seq_num):
            seq_len = seq_lens[seq_i]
            block_max_len = 0
            for layer_i in range(seq_len):
                if layer_i <= max_window:
                    block_max_len += (seq_len - layer_i) * (layer_i + 1)
                else:
                    block_max_len += (max_window + 1) * max_window
            self.block_sizes[seq_i] = block_max_len

    def gather(self, indices, cache_ids):
        # Gather tensors according to CacheItem pairs
        tensors_gathered = []
        if isinstance(indices, torch.Tensor):
            gather_indices = indices
        else:
            gather_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        for cache_id in cache_ids:
            tensor_block = self.caches[cache_id]
            tensor_gather = tensor_block.index_select(dim=0, index=gather_indices)
            tensors_gathered.append(tensor_gather)
        return tensors_gathered

    def fill(self, cache_id_offset, cache_id_len, cache_ids, values):
        if len(cache_ids) != len(values):
            raise Exception('TensorCache::fill names and values mismatch')
        # cdef PyObject ** tensor_block
        for cache_id, value in zip(cache_ids, values):
            tensor_block = self.caches[cache_id]
            tensor_block[cache_id_offset: cache_id_offset + cache_id_len] = value
            if self.cache_types[cache_id] == CacheType.DETACH:
                detach_offset = self.total_block_size + cache_id_offset
                tensor_block[detach_offset: detach_offset + cache_id_len] = value.detach()

    def get(self, cache_id, idx):
        assert self.caches[cache_id] is not None
        return self.caches[cache_id][idx]

    def detach(self, idx):
        detach_idx = self.total_block_size + idx
        for cache_i, cache_type in enumerate(self.cache_types):
            if cache_type == CacheType.DETACH:
                self.caches[cache_i][detach_idx] = self.caches[cache_i][idx].detach()
        return detach_idx