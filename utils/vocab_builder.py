import codecs
from math import ceil, log
import os
import torch
from collections import deque
import numpy as np
from typing import List, Dict
from utils.tree_utils import get_tree_from_merge_trajectory
from tqdm import tqdm
import pickle


TOKEN_SEP = '\0'
SEG_SEP = '\1'


class WordTreeNode:
    def __init__(self, id, is_word, prev_node=None):
        self.is_word = is_word
        # self.entries = {}
        self.entries = {}
        self.total = 0
        self.delta_loss = 0
        self.prev_node = prev_node
        self.score = float('inf')
        self.id = id

    def add(self, ids: List[int], cnt):
        if len(ids) == 0:
            self.is_word = True
            self.total += cnt
        else:
            next_entry = self.entries.get(ids[0], WordTreeNode(ids[0], False, prev_node=self))
            next_entry.add(ids[1:], cnt)
            self.entries[ids[0]] = next_entry
    
    def add_delta_loss(self, ids: List[int], delta):
        if len(ids) == 0:
            assert self.is_word
            self.delta_loss += delta
        else:
            next_entry = self.entries.get(ids[0], None)
            assert next_entry is not None
            next_entry.add_delta_loss(ids[1:], delta)

    def has(self, ids: List[int]):
        if len(ids) == 0:
            return self.is_word
        else:
            next_entry = self.entries.get(ids[0], None)
            if next_entry is not None:
                return next_entry.has(ids[1:])
            else:
                return False

    def update_scores(self, total):
        if self.is_word:
            if self.total > 0:
                self.score = -log(self.total / total)
            else:
                self.score = float('inf')
        for next_entry in self.entries.values():
            next_entry.update_scores(total)

    def get_score(self, ids):
        if len(ids) == 0:
            if self.is_word:
                return self.score
            else:
                return None
        else:
            next_entry = self.entries.get(ids[0], None)
            if next_entry is not None:
                return next_entry.get_score(ids[1:])
            else:
                return None

    def get_count(self, ids):
        if len(ids) == 0:
            if self.is_word:
                return self.total
            else:
                return None
        else:
            next_entry = self.entries.get(ids[0], None)
            if next_entry is not None:
                return next_entry.get_count(ids[1:])
            else:
                return None

    def merge(self, other):
        self.is_word = self.is_word or other.is_word
        if self.is_word:
            self.total += other.total
        for wid, entry in other.entries.items():
            my_node = self.entries.get(wid, WordTreeNode(wid, False, self))
            my_node.merge(entry)
            self.entries[wid] = my_node

    def remove(self, id):
        if id in self.entries:
            self.entries.pop(id)
        if len(self.entries) == 0 and not self.is_word:
            if self.prev_node is not None:
                self.prev_node.remove(self.id)

    def prune_by(self, condition_fn):
        if len(self.entries) == 0 and not self.is_word:
            # checkpath
            if self.prev_node is not None:
                self.prev_node.remove(self.id)
        for child in self.entries.values():
            if child.is_word and condition_fn(child):
                child.is_word = False
                child.total = 0
                child.delta_loss = 0
            child.prune_by(condition_fn)

    def recursive_apply(self, fn):
        if self.is_word:
            fn(self)
        for entry in self.entries.values():
            entry.recursive_apply(fn)

    def yield_word(self, prev_ids=[]):
        if len(prev_ids) > 0 and self.is_word:
            yield prev_ids + [self.id], self.total
        for entry in self.entries.values():
            yield from entry.yield_word(prev_ids + [self.id])

class WordTree:
    def __init__(self, basic_vocab_size):
        self.basic_vocab_size = basic_vocab_size
        self.basic_entries = [WordTreeNode(wid, True) for wid in range(basic_vocab_size)]
        self.total = 0

    def add(self, ids: List[int], cnt=1):
        # count + 1
        self.total += cnt
        assert len(ids) >= 1
        assert 0 <= ids[0] < len(self.basic_entries)
        self.basic_entries[ids[0]].add(ids[1:], cnt)

    def add_delta_loss(self, ids: List[int], delta):
        # count + 1
        assert len(ids) >= 1
        assert 0 <= ids[0] < len(self.basic_entries)
        self.basic_entries[ids[0]].add_delta_loss(ids[1:], delta)

    def has(self, ids: List[int]):
        assert len(ids) >= 1
        assert 0 <= ids[0] < len(self.basic_entries) 
        return self.basic_entries[ids[0]].has(ids[1:])

    def update_scores(self):
        for entry in self.basic_entries:
            entry.update_scores(self.total)

    def get_score(self, ids: List[int]):
        assert 0 <= ids[0] < len(self.basic_entries) 
        return self.basic_entries[ids[0]].get_score(ids[1:])

    def get_count(self, ids: List[int]):
        assert 0 <= ids[0] < len(self.basic_entries) 
        return self.basic_entries[ids[0]].get_count(ids[1:])
    
    def merge(self, other: WordTreeNode):
        self.total += other.total
        assert len(self.basic_entries) == len(other.basic_entries)
        for entry, other_entry in zip(self.basic_entries, other.basic_entries):
            entry.merge(other_entry)

    def vocab_iterator(self):
        for entry in self.basic_entries:
            yield from entry.yield_word()

    def truncate_by_count(self, threshold):
        # gather all totals
        for entry in self.basic_entries:
            entry.prune_by(lambda x: x.total < threshold)

        self.total = 0
        def cumsum(x):
            self.total += x.total

        for entry in self.basic_entries:
            entry.recursive_apply(cumsum)

    def truncate_by_delta_loss(self, count):
        # gather all totals
        delta_loss = []
        for entry in self.basic_entries:
            entry.recursive_apply(lambda x: delta_loss.append(x.delta_loss))

        delta_loss.sort(reverse=True)
        count = min(count, len(delta_loss) - 1)
        threshold = delta_loss[count]
        for entry in self.basic_entries:
            entry.prune_by(lambda x: x.delta_loss <= threshold)

        # refresh total sum
        self.total = 0
        def cumsum(x):
            self.total += x.total

        for entry in self.basic_entries:
            entry.recursive_apply(cumsum)

    def reset(self):
        # reset count and delta_loss
        def reset_count_d_loss(node):
            node.total = 0
            node.delta_loss = 0
        
        for entry in self.basic_entries:
            entry.recursive_apply(reset_count_d_loss)

    def size(self):
        cnt = 0
        def count(x):
            nonlocal cnt
            if x.is_word:
                cnt += 1

        for entry in self.basic_entries:
            entry.recursive_apply(count)

        return max(0, cnt - self.basic_vocab_size)


class WordNode:
    def __init__(self, i, j, node) -> None:
        self.i = i
        self.j = j
        
        self.node = node
        self.left = None
        self.right = None
        self._tokens = None
        self._segments = None
        self.hit_vocab = False

    def __str__(self):
        if self.i == self.j:
            return ''.join(self._tokens)
        return f'({self.left} {self.right})'

    def tokens_and_segments(self, ids, vocab, external_vocab, subword_prefix=None):
        if self._segments is None and self._tokens is None:
            if self.left is None and self.right is None:
                assert self.node.i == self.node.j
                self._tokens = [vocab[ids[self.node.i]]]
                self._segments = [vocab[ids[self.node.i]]]
            else:
                _, left_tokens = self.left.tokens_and_segments(ids, vocab, external_vocab)
                _, right_tokens = self.right.tokens_and_segments(ids, vocab, external_vocab)
                self._segments = left_tokens + right_tokens
                if TOKEN_SEP.join(self._segments) in external_vocab:
                    self.hit_vocab = True
                    self._tokens = [TOKEN_SEP.join(self._segments)]  # merge into whole word
                else:
                    self._tokens = self._segments
        # a token is an entry in the basic vocabulary and the external vocabulary
        # a semgent is a combination of two tokens
        return self._segments, self._tokens
        

def count_segments(pair_path, basic_vocab, external_vocab, subword_prefix=None):
    counter = {}
    left_counter = {}
    right_counter = {}
    ngram_segments = {}
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
    n_sent = 0
    # for indices, ids in tqdm(indices_ids_pairs):
    offset = 0
    for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
        # mark span to whole word tree
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]
        offset += ids_len
        if len(ids) == 0:
            continue
        root = get_tree_from_merge_trajectory(indices, len(ids))
        word_tree_root = convert_tree_to_wordtree(root)

        # count ngram over words
        queue = deque()
        queue.append(word_tree_root)
        while len(queue) > 0:
            current_word_node = queue.popleft()
            segments, tokens = current_word_node.tokens_and_segments(
                ids, basic_vocab, external_vocab, subword_prefix=subword_prefix)
            
            if len(segments) == 2:
                # add segment count
                counter.setdefault(TOKEN_SEP.join(segments), 0)
                counter[TOKEN_SEP.join(segments)] += 1

                if not current_word_node.hit_vocab:
                    # a potential bi-gram candidate
                    segment_key = TOKEN_SEP.join(segments)
                    segment_set = ngram_segments.get(segment_key, set())
                    segment_set.add(SEG_SEP.join(segments))
                    ngram_segments[segment_key] = segment_set
                # add l_r, r_l
                right_counter.setdefault(segments[0], {})
                left_counter.setdefault(segments[1], {})
                right_counter[segments[0]].setdefault(segments[1], 0)
                left_counter[segments[1]].setdefault(segments[0], 0)
                right_counter[segments[0]][segments[1]] += 1
                left_counter[segments[1]][segments[0]] += 1
            elif len(tokens) == 1:
                counter.setdefault(tokens[0], 0)
                counter[tokens[0]] += 1
            if current_word_node.left is not None and current_word_node.right is not None \
                and not current_word_node.hit_vocab:  # stop if hit vocab
                queue.append(current_word_node.left)
                queue.append(current_word_node.right)
    return counter, left_counter, right_counter, ngram_segments


def build_vocab(external_vocab, counter, left_counter, right_counter, ngram_segments, keep_size=100000, count_threshold=3):
    left_entropy = {}
    for key in left_counter.keys():
        # print(list(left_counter[key].values()))
        left_cnt = np.array(list(left_counter[key].values()))
        total = left_cnt.sum()
        left_dist = left_cnt / total
        left_entropy[key] = (-np.log(left_dist) * left_dist).sum()
    right_entropy = {}
    for key in right_counter.keys():
        right_cnt = np.array(list(right_counter[key].values()))
        total = right_cnt.sum()
        right_dist = right_cnt / total
        right_entropy[key] = (-np.log(right_dist) * right_dist).sum()

    total_count = sum(counter.values())
    pmi_table = {}
    for ngram, segments in ngram_segments.items():
        min_pmi = np.inf
        if counter[ngram] < count_threshold:
            continue
        for seg in segments:
            parts = seg.split(SEG_SEP)
            assert len(parts) == 2

            p_ = (total_count / counter[parts[0]]) * (total_count / counter[parts[1]])
            h_r_l = left_entropy[parts[1]]
            h_l_r = right_entropy[parts[0]]
            min_pmi = min(min_pmi, p_ / (total_count / counter[ngram] - min(h_r_l, h_l_r)))
        pmi_table[ngram] = min_pmi
    for ngram, pmi in external_vocab.items():
        if ngram in counter and counter[ngram] >= count_threshold:
            pmi_table[ngram] = min(pmi, pmi_table.get(ngram, np.inf))

    sorted_pmi = sorted(pmi_table.items(), key=lambda x: x[1], reverse=True)
    new_vocab = {}
    for key, pmi in sorted_pmi[:keep_size]:
        new_vocab[key] = pmi
    return new_vocab


def convert_tree_to_wordtree(root):
    # mark span to whole word tree
    queue = deque()
    word_tree_root = WordNode(root.i, root.j, root)
    queue.append((root, word_tree_root))
    while len(queue) > 0:
        current, word_node = queue.popleft()
        left, right = current.left, current.right
        if left is not None and right is not None:
            left_word_node = WordNode(left.i, left.j, left)
            right_word_node = WordNode(right.i, right.j, right)
            queue.append((left, left_word_node))
            queue.append((right, right_word_node))
            word_node.left = left_word_node
            word_node.right = right_word_node
    return word_tree_root

def load_dict(dict_path):
    external_vocab = {}
    vocab_id = 1
    with codecs.open(dict_path, mode='r') as f:
        for line in f:
            key = line.strip()
            external_vocab[key] = vocab_id
            vocab_id += 1
    return external_vocab

def load_span_tokenizer(external_vocab_path):
    import cppbackend
    external_vocab = []
    max_vocab_id = -1
    with codecs.open(external_vocab_path, mode='r') as f:
        for line in f:
            if '\t' in line:
                line = line.split('\t')[0]
            ids = line.split(',')
            ids = list(map(lambda x: int(x), ids))
            ids = np.array(ids)
            max_vocab_id = max(max_vocab_id, max(ids))
            external_vocab.append(ids)
    return cppbackend.SpanTokenizer(external_vocab, max_vocab_id + 1)

# def unigram_E_step(pair_path, vocab_model):
#     with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
#         pair_lens = pickle.load(f_in)
#     memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')


#     def best_segment(root, input_ids):
#         if root.i == root.j:
#             score = vocab_model.get_score(input_ids[root.i: root.j + 1])
#             return score, [input_ids[root.i: root.j + 1]]
#         else:
#             left_score, left_tokens = best_segment(root.left, input_ids)
#             right_score, right_tokens = best_segment(root.right, input_ids)
#             hit_score = vocab_model.get_score(input_ids[root.i: root.j + 1])
#             if hit_score is not None:
#                 if left_score + right_score > hit_score:
#                     return hit_score, [input_ids[root.i: root.j + 1]]

#             return left_score + right_score, left_tokens + right_tokens

#     total_score = 0
#     # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
#     offset = 0
#     for pair_i in tqdm(range(len(pair_lens) // 2)):
#         indices_len = pair_lens[pair_i * 2]
#         ids_len = pair_lens[pair_i * 2 + 1]
#         indices = memmap[offset: offset + indices_len]
#         offset += indices_len
#         ids = memmap[offset: offset + ids_len]
#         offset += ids_len
#         if len(ids) == 0:
#             continue
        
#         root = get_tree_from_merge_trajectory(indices, len(ids))
#         score, segments = best_segment(root, ids)

#         for segment in segments:
#             vocab_model.add(segment)

#         # if pair_i > 10000:
#         #     break

# def unigram_M_step(pair_path, vocab_model):
#     with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
#         pair_lens = pickle.load(f_in)
#     memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')


#     def delta_loss(root, input_ids, delta_loss_record):
#         if root.i == root.j:
#             score = vocab_model.get_score(input_ids[root.i: root.j + 1])
#             return score, [root]
#         else:
#             left_score, left_nodes = delta_loss(root.left, input_ids, delta_loss_record)
#             right_score, right_nodes = delta_loss(root.right, input_ids, delta_loss_record)
#             hit_score = vocab_model.get_score(input_ids[root.i: root.j + 1])
#             if hit_score is not None:
#                 if left_score + right_score > hit_score:
#                     delta_loss_record[root] = left_score + right_score - hit_score
#                     # vocab_model.add_delta_loss(input_ids[root.i: root.j + 1], left_score + right_score - hit_score)
#                     return hit_score, [root]

#             return left_score + right_score, left_nodes + right_nodes

#     total_score = 0
#     offset = 0
#     # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
#     for pair_i in tqdm(range(len(pair_lens) // 2)):
#         indices_len = pair_lens[pair_i * 2]
#         ids_len = pair_lens[pair_i * 2 + 1]
#         indices = memmap[offset: offset + indices_len]
#         offset += indices_len
#         ids = memmap[offset: offset + ids_len]
#         offset += ids_len
#         if len(ids) == 0:
#             continue
        
#         root = get_tree_from_merge_trajectory(indices, len(ids))
#         delta_loss_record = {}
#         _, seg_nodes = delta_loss(root, ids, delta_loss_record)
#         for node in seg_nodes:
#             if node.i != node.j:
#                 d_loss = delta_loss_record[node]
#                 vocab_model.add_delta_loss(ids[node.i: node.j + 1], d_loss)

        # if pair_i > 10000:
        #     break


def count_basic_vocab(pair_path, vocab_model):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
    offset = 0
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]
        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        to_visit = [root]
        while len(to_visit) > 0:
            top = to_visit.pop(-1)

            if top.left is not None and top.right is not None:
                to_visit.append(top.right)
                to_visit.append(top.left)
            else:
                vocab_model.add(ids[top.i:top.j + 1])
        
        # if pair_i > 10000:
        #     break
    return vocab_model

def count_bigrams(pair_path, vocab_model):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
    offset = 0
    new_vocab_model = WordTree(len(vocab_model.basic_entries))
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):

    def recursive_count_bigram(node, ids):
        if node.left is not None and node.right is not None:
            left_hit = recursive_count_bigram(node.left, ids)
            right_hit = recursive_count_bigram(node.right, ids)
            if left_hit and right_hit:
                # check if self is hit
                if vocab_model.has(ids[node.i : node.j + 1]):
                    return True
                else:
                    new_vocab_model.add(ids[node.i : node.j + 1])
                    return False
            else:
                return False
        else:
            return True

    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]


        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        recursive_count_bigram(root, ids)
    
        # if pair_i > 10000:
        #     break
    return new_vocab_model
    # new_vocab_model.truncate_by_count(threshold)
    # affected_ids = set()
    # for ids, count in new_vocab_model.vocab_iterator():
    #     for w_id in ids:
    #         affected_ids.add(w_id)

    # vocab_model.merge(new_vocab_model)
    # return affected_ids

# def _to_ids_str(ids):
#     return ','.join([str(_) for _ in ids])

# def count_basic_vocab_dict(pair_path):
#     with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
#         pair_lens = pickle.load(f_in)
#     memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
#     offset = 0
#     vocab_model = {}
#     # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
#     for pair_i in tqdm(range(len(pair_lens) // 2)):
#         indices_len = pair_lens[pair_i * 2]
#         ids_len = pair_lens[pair_i * 2 + 1]
#         indices = memmap[offset: offset + indices_len]
#         offset += indices_len
#         ids = memmap[offset: offset + ids_len]
#         offset += ids_len
#         if len(ids) == 0:
#             continue
        
#         root = get_tree_from_merge_trajectory(indices, len(ids))
#         to_visit = [root]
#         while len(to_visit) > 0:
#             top = to_visit.pop(-1)

#             if top.left is not None and top.right is not None:
#                 to_visit.append(top.right)
#                 to_visit.append(top.left)
#             else:
#                 key = _to_ids_str()
#                 count = vocab_model.get()
#                 vocab_model[ids[top.i:top.j + 1]]
        
#         # if pair_i > 10000:
#         #     break
#     return vocab_model

def to_ids_key(ids):
    return ','.join([str(_) for _ in ids])

def count_basic_vocab_dict(pair_path):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
    offset = 0
    counter = {}
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]
        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        to_visit = [root]
        while len(to_visit) > 0:
            top = to_visit.pop(-1)

            if top.left is not None and top.right is not None:
                to_visit.append(top.right)
                to_visit.append(top.left)
            else:
                key = to_ids_key(ids[top.i:top.j + 1])
                prev_count = counter.get(key, 0)
                counter[key] = prev_count + 1
        
        # if pair_i > 10000:
        #     break
    return counter


def count_bigrams_dict(pair_path, basic_vocab, new_vocab):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')
    offset = 0
    new_vocab_model = {}
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):

    def recursive_count_bigram(node, ids):
        if node.left is not None and node.right is not None:
            left_hit, is_left_new = recursive_count_bigram(node.left, ids)
            right_hit, is_right_new  = recursive_count_bigram(node.right, ids)
            
            if left_hit and right_hit:
                # check if self is hit
                ids_key = to_ids_key(ids[node.i : node.j + 1])
                if (is_left_new or is_right_new) and ids_key not in basic_vocab:
                    # not in basic_vocab and new_vocab
                    prev_count = new_vocab_model.get(ids_key, 0)
                    new_vocab_model[ids_key] = prev_count + 1
                return ids_key in basic_vocab, ids_key in new_vocab
            else:
                return False, False
        else:
            ids_key = to_ids_key(ids[node.i : node.j + 1])
            return True, ids_key in new_vocab

    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]


        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        recursive_count_bigram(root, ids)
    
        # if pair_i > 10000:
        #     break
    del memmap
    return new_vocab_model


def unigram_E_step(pair_path, vocab_model):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')

    count_dict = {}
    def best_segment(root, input_ids):
        if root.i == root.j:
            key = to_ids_key(input_ids[root.i: root.j + 1])
            # assert key in vocab_model
            score = vocab_model.get(key, float('inf'))
            return score, [input_ids[root.i: root.j + 1]]
        else:
            left_score, left_tokens = best_segment(root.left, input_ids)
            right_score, right_tokens = best_segment(root.right, input_ids)
            key = to_ids_key(input_ids[root.i: root.j + 1])
            hit_score = vocab_model.get(key, float('inf'))
            if left_score + right_score > hit_score:
                return hit_score, [input_ids[root.i: root.j + 1]]

            return left_score + right_score, left_tokens + right_tokens

    total_score = 0
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
    offset = 0
    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]
        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        score, segments = best_segment(root, ids)

        for segment in segments:
            key = to_ids_key(segment)
            count_dict.setdefault(key, 0)
            count_dict[key] += 1
        # if pair_i > 1000:
        #     break
    return count_dict

def unigram_M_step(pair_path, vocab_scores: Dict[str, float]):
    with open(f'{pair_path}.len.pkl', mode='rb') as f_in:
        pair_lens = pickle.load(f_in)
    memmap = np.memmap(pair_path, dtype=np.int32, mode='r', order='C')

    def delta_loss(root, input_ids, delta_loss_record):
        if root.i == root.j:
            key = to_ids_key(input_ids[root.i: root.j + 1])
            # assert key in vocab_scores
            score = vocab_scores.get(key, float('inf'))
            return score, [root]
        else:
            left_score, left_nodes = delta_loss(root.left, input_ids, delta_loss_record)
            right_score, right_nodes = delta_loss(root.right, input_ids, delta_loss_record)
            key = to_ids_key(input_ids[root.i: root.j + 1])
            hit_score = vocab_scores.get(key, float('inf'))
            if left_score + right_score > hit_score:
                delta_loss_record[root] = left_score + right_score - hit_score
                # vocab_model.add_delta_loss(input_ids[root.i: root.j + 1], left_score + right_score - hit_score)
                return hit_score, [root]

            return left_score + right_score, left_nodes + right_nodes

    total_score = 0
    offset = 0
    # for indices_len, ids_len in tqdm(zip(pair_lens[::2], pair_lens[1::2])):
    d_loss_sum = {}
    for pair_i in tqdm(range(len(pair_lens) // 2)):
        indices_len = pair_lens[pair_i * 2]
        ids_len = pair_lens[pair_i * 2 + 1]
        indices = memmap[offset: offset + indices_len]
        offset += indices_len
        ids = memmap[offset: offset + ids_len]
        offset += ids_len
        if len(ids) == 0:
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        delta_loss_record = {}
        _, seg_nodes = delta_loss(root, ids, delta_loss_record)
        for node in seg_nodes:
            if node.i != node.j:
                d_loss = delta_loss_record[node]
                # vocab_model.add_delta_loss(ids[node.i: node.j + 1], d_loss)
                key = to_ids_key(ids[node.i: node.j + 1])
                d_loss_sum.setdefault(key, 0)
                d_loss_sum[key] += d_loss
        # if pair_i > 1000:
        #     break
    return d_loss_sum