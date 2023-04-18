from model.r2d2_common import CacheSlots


class PyNode:
    def __init__(self, left, right, i, j, cache_id) -> None:
        self._i = i
        self._j = j
        self._cache_id = cache_id
        self._decode_cache_id = -1
        self._left = left
        self._right = right
        self.label = None

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @i.setter
    def i(self, v):
        self._i = v

    @j.setter
    def j(self, v):
        self._j = v

    @property
    def pos(self):
        return self._i

    @property
    def is_leaf(self):
        return self._left is None and self._right is None

    @property
    def cache_id(self):
        return self._cache_id

    @property
    def decode_cache_id(self):
        return self._decode_cache_id

    @decode_cache_id.setter
    def decode_cache_id(self, val):
        self._decode_cache_id = val

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right


class PyCell:
    def __init__(self, i, j, best_node_idx, beams, tensor_cache) -> None:
        self.token_id = -1
        self._beams = beams
        self._i = i
        self._j = j
        self._tensor_cache = tensor_cache
        self._best_k = best_node_idx

    @property
    def beam_size(self):
        return len(self._beams)

    @property
    def cache_id_offset(self):
        return self._beams[0].cache_id

    @property
    def pos(self):
        return self._i

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @property
    def is_leaf(self):
        return self._i == self._j

    @property
    def e_ij(self):
        return self._tensor_cache.get(CacheSlots.E_IJ, self._beams[self._best_k].cache_id)

    @property
    def best_node(self):
        return self._beams[self._best_k]


class PyTable:
    def __init__(self, root, seq_len, beam_size) -> None:
        self._seq_len = seq_len
        self._root = root
        self._beam_size = beam_size

    @property
    def root(self):
        return self._root

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def beam_size(self):
        return self._beam_size