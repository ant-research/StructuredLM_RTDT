import codecs
from math import ceil
import os
import torch
from collections import deque
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoTokenizer
from model.topdown_parser import LSTMParser
from reader.memory_line_reader import BatchSelfRegressionLineDataset
from utils.model_loader import load_model
from torch.utils.data import DataLoader, SequentialSampler
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory


TOKEN_SEP = '\0'
SEG_SEP = '\1'


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

    def __mergable(self, segments, subword_prefix):
        if subword_prefix is not None and  len(segments) == 2 \
            and segments[1].startswith(subword_prefix):
            return True
        return False

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
                if TOKEN_SEP.join(self._segments) in external_vocab or \
                    self.__mergable(self._segments, subword_prefix):
                    self.hit_vocab = True
                    self._tokens = [TOKEN_SEP.join(self._segments)]  # merge into whole word
                else:
                    self._tokens = self._segments
        return self._segments, self._tokens
        

def build_vocab_master(inputs, basic_vocab, external_vocab, subword_prefix=None):
    counter = {}
    ngram_segments = {}
    n_sent = 0
    for root, ids in inputs:
        # mark span to whole word tree
        word_tree_root = convert_tree_to_wordtree(root)

        # count ngram over words
        queue = deque()
        queue.append(word_tree_root)
        while len(queue) > 0:
            current_word_node = queue.popleft()
            segments, tokens = current_word_node.tokens_and_segments(
                ids, basic_vocab, external_vocab, subword_prefix=subword_prefix)
            if len(segments) == 2:
                counter.setdefault(TOKEN_SEP.join(segments), 0)
                counter[TOKEN_SEP.join(segments)] += 1

                if not current_word_node.hit_vocab:
                    segment_key = TOKEN_SEP.join(segments)
                    segment_set = ngram_segments.get(segment_key, set())
                    segment_set.add(SEG_SEP.join(segments))
                    ngram_segments[segment_key] = segment_set
            elif len(tokens) == 1:
                counter.setdefault(tokens[0], 0)
                counter[tokens[0]] += 1
            if current_word_node.left is not None and current_word_node.right is not None \
                and not current_word_node.hit_vocab:  # stop if hit vocab
                queue.append(current_word_node.left)
                queue.append(current_word_node.right)
        n_sent += 1
        if n_sent % 1000 == 0:
            segments = word_tree_root._segments
    return counter, ngram_segments


def merge_counter_and_segments(counter_list, ngram_segments_list):
    all_counter = {}
    all_ngram_segments = {}
    for counter, ngram_segments in zip(counter_list, ngram_segments_list):
        for word, count in counter.items():
            all_counter[word] = counter.get(word, 0) + count
        for ngram, segments in ngram_segments.items():
            segments_set = all_ngram_segments.get(ngram, set())
            segments_set.update(segments)
            all_ngram_segments[ngram] = segments_set
    return all_counter, all_ngram_segments


def build_vocab(external_vocab, counter, ngram_segments, keep_size=100000, count_threshold=3):
    total_count = sum(counter.values())
    pmi_table = {}
    for ngram, segments in ngram_segments.items():
        min_pmi = np.inf
        if counter[ngram] < count_threshold:
            continue
        for seg in segments:
            parts = seg.split(SEG_SEP)
            p_ = 1
            for part in parts:
                p_ *= total_count / counter[part]
            min_pmi = min(min_pmi, p_ / (total_count / counter[ngram]))
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


def build_vocab_for_corpus(parser_path, corpus_path, config_dir, vocab_output_path, \
                           keep_size=1000, count_threshold=2, stop_threshold=0.95):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(os.path.join(config_dir, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(config_dir)
    parser = LSTMParser(config)
    load_model(parser, parser_path)
    parser.to(device)
    dataset = BatchSelfRegressionLineDataset(
        corpus_path,
        tokenizer,
        batch_max_len=16384,
        min_len=2,
        batch_size=50,
        max_line=-1,
        input_type="txt",
        random=False
    )

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=SequentialSampler(dataset),
                            collate_fn=dataset.collate_batch)
    epoch_iterator = tqdm(dataloader, desc="Iterat ion")

    tree_ids_pairs = []
    for step, inputs in enumerate(epoch_iterator):
        with torch.no_grad():
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            merge_trajectories = parser(**inputs)
            s_indices = merge_trajectories.cpu().data.numpy()
            ids_np = inputs['input_ids'].cpu().data.numpy()
            seq_lens = inputs['attention_mask'].sum(dim=-1).cpu().data.numpy()
            for sent_i, seq_len in enumerate(seq_lens):
                root = get_tree_from_merge_trajectory(s_indices[sent_i], seq_len)
                tree_ids_pairs.append([root, ids_np[sent_i, :seq_len]])

    subword_prefix = '##'
    basic_vocab = [None] * len(tokenizer.vocab)
    for key, idx in tokenizer.vocab.items():
        basic_vocab[idx] = key
    external_vocab = dict()
    while True:
        print('count segments')
        counter, ngram_segments = build_vocab_master(tree_ids_pairs, basic_vocab, external_vocab, 
                                                     subword_prefix=subword_prefix)
        print('start building vocab')
        new_vocab = build_vocab(external_vocab, counter, ngram_segments, 
                                keep_size=keep_size, count_threshold=count_threshold)

        
        keep_rate = len(new_vocab.keys() & external_vocab.keys()) / keep_size
        external_vocab = new_vocab
        print(f'keep rate: {keep_rate}')
        if keep_rate >= stop_threshold:
            break

    with codecs.open(vocab_output_path, mode='w', encoding='utf-8') as f_out:
        for key, pmi in external_vocab.items():
            print(f'{key}\t{pmi}', file=f_out)


def tokenizing_with_custom_vocab(parser_path, config_dir, vocab_path, corpus_path, output_path):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(os.path.join(config_dir, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(config_dir)
    parser = LSTMParser(config)
    print('load model')
    load_model(parser, parser_path)
    parser.to(device)
    print('load data')
    dataset = BatchSelfRegressionLineDataset(
        corpus_path,
        tokenizer,
        batch_max_len=16384,
        min_len=2,
        batch_size=50,
        max_line=-1,
        input_type="txt",
        random=False
    )

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=SequentialSampler(dataset),
                            collate_fn=dataset.collate_batch)
    epoch_iterator = tqdm(dataloader, desc="Iterat ion")
    basic_vocab = [None] * len(tokenizer.vocab)
    for key, idx in tokenizer.vocab.items():
        basic_vocab[idx] = key
    external_vocab = dict()
    print('load dict')
    with codecs.open(vocab_path, mode='r', encoding='utf-8') as f_in:
        for line in f_in:
            parts = line.strip().split('\t')
            external_vocab[parts[0]] = float(parts[1])
    tree_ids_pairs = []
    with codecs.open(output_path, mode='w', encoding='utf-8') as f_out:
        for _, inputs in enumerate(epoch_iterator):
            with torch.no_grad():
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                merge_trajectories = parser(**inputs)
                s_indices = merge_trajectories.cpu().data.numpy()
                ids_np = inputs['input_ids'].cpu().data.numpy()
                seq_lens = inputs['attention_mask'].sum(dim=-1).cpu().data.numpy()
                for sent_i, seq_len in enumerate(seq_lens):
                    root = get_tree_from_merge_trajectory(s_indices[sent_i], seq_len)
                    tree_ids_pairs.append([root, ids_np[sent_i, :seq_len]])
                    word_tree_root = convert_tree_to_wordtree(root)
                    segments, _ = word_tree_root.tokens_and_segments(ids_np[sent_i, :seq_len], basic_vocab, external_vocab)
                    segments = [val.replace(TOKEN_SEP, '') for val in segments]
                    print(u' '.join(segments), file=f_out)

if __name__ == '__main__':
    build_vocab_for_corpus('data/cn_wiki_pretrain/parser9.bin', 'data/key_word_mining/raw_text.txt', \
                           'data/cn_wiki_pretrain', 'data/key_word_mining/vocab_mining_5.txt',
                           keep_size=20000, count_threshold=5,\
                           stop_threshold=0.99)
    # tokenizing_with_custom_vocab('data/save/cnwiki_4l/parser.bin', 'data/save/cnwiki_4l',\
    #                              'data/licaishi/vocab_mining.txt', 'data/licaishi/lcs_raw.txt',
    #                              'data/licaishi/lcs_tokenized.txt')