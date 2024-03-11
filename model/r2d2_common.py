from collections import namedtuple


class CacheSlots:
    E_IJ = 0
    LOG_P_IJ_SUM = 1
    NT_SCORE = 2


LMLossParam = namedtuple(
    'LMLossParam',
    [
        'model',
        'chart_tables',
        'tensor_cache',
        'input_ids',
        'flatten_input_ids',
        's_indices',
        'atom_spans',
        'seq_lens'
    ]
)

NodeCombination = namedtuple(
    'NodeCombination',
    [
        'node_ik',
        'node_kj',
        'left',
        'right'
    ]
)

BOS_CACHE_ID = 0
EOS_CACHE_ID = 1
INF_LOG_P_ID = 2
SPECIAL_TOKEN_NUM = 3  # BOS, EOS

ROLE_LEFT = 1
ROLE_RIGHT = 2
ROLE_PARENT = 3
