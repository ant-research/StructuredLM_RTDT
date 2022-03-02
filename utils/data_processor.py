import codecs
from random import random
import re
from utils.misc import _align_spans, get_sentence_from_words
import numpy as np
import argparse
import sys


def split_wiki_corpus(input_path, output_path, split_pattern, remove_puncts):
    with codecs.open(input_path, mode='r', encoding='utf-8') as in_file, \
            codecs.open(output_path, mode='w', encoding='utf-8') as out_file:
        for line in in_file:
            if line.startswith('http'):
                continue
            sentences = re.split(split_pattern, line.strip())
            sentences.append('')
            sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
            for s in sentences:
                if len(s) > 0:
                    for punct_pattern in remove_puncts:
                        s = s.replace(punct_pattern, ' ')
                    # print(s)
                    print(s, file=out_file)


def convert_txt_to_ids(path, tokenizer, output_path, max_len=32):
    total_sentences = 0
    discard_sentences = 0
    with codecs.open(path, mode='r', encoding='utf-8') as f, \
            codecs.open(output_path, mode='w', encoding='utf-8') as out_f:
        for line in f:
            if len(line.strip()) > 0:
                tokens = tokenizer.tokenize(line)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(ids) < max_len:
                    print(' '.join([str(_) for _ in ids]), file=out_f)
                    total_sentences += 1
                else:
                    discard_sentences += 1
                if total_sentences % 100 == 0:
                    print(f'all sentence: {total_sentences}, discard rate: {discard_sentences / total_sentences}')


def convert_txt_to_ids_spans(path, tokenizer, seperator, output_path, max_len=32):
    total_sentences = 0
    discard_sentences = 0
    with codecs.open(path, mode='r', encoding='utf-8') as f, \
            codecs.open(output_path, mode='w', encoding='utf-8') as out_f:
        for line in f:
            if len(line.strip()) > 0:
                sentence, spans = get_sentence_from_words(line.strip().split(seperator), seperator)
                outputs = tokenizer.encode_plus(sentence, add_special_tokens=False, return_offsets_mapping=True)
                new_spans = outputs['offset_mapping']
                word_starts, word_ends = _align_spans(spans, new_spans)
                atom_spans = []
                for st, ed in zip(word_starts, word_ends):
                    if st != ed:
                        atom_spans.append([st, ed])
                tokens = tokenizer.tokenize(line)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(ids) < max_len:
                    ids_str = ' '.join([str(_) for _ in ids])
                    spans_str = ';'.join([f'{span[0]},{span[1]}' for span in atom_spans])
                    print(f"{ids_str}|{spans_str}", file=out_f)
                    total_sentences += 1
                else:
                    discard_sentences += 1
            if total_sentences % 100 == 0:
                print(f'all sentence: {total_sentences}, discard rate: {discard_sentences / total_sentences}')


def avg_length_statistics(path):
    total_token = 0
    total_items = 0
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                ids_str = None
                if ';' in line:
                    parts = line.split(';')
                    ids_str = parts[0]
                else:
                    ids_str = line
                total_token += len(ids_str.split())
                total_items += 1
    print(f'total token: {total_token}, avg len: {total_token / total_items}')


def random_select_sentences(path, output_path, ranges, size_per_range):
    items = []
    with codecs.open(path, mode='r', encoding='utf-8') as f_in:
        for line in f_in:
            ids = line.split()
            items.append(ids)
    np.random.shuffle(items)
    buckets = [[] for _ in range(len(ranges))]
    collected = 0
    expected_collected = len(ranges) * size_per_range
    for ids in items:
        for bucket_id, sent_len in enumerate(ranges):
            if collected == expected_collected:
                break
            if len(ids) < sent_len:
                if len(buckets[bucket_id]) < size_per_range:
                    buckets[bucket_id].append(ids)
                    collected += 1
                break
    for range_id, sent_len in enumerate(ranges):
        if range_id == 0:
            lower = 0
        else:
            lower = ranges[range_id - 1]
        with codecs.open(f'{output_path}.{sent_len}', mode='w', encoding='utf-8') as f_out:
            for ids_list in buckets:
                for ids in ids_list:
                    if len(ids) >= lower and len(ids) < sent_len:
                        print(' '.join(ids), file=f_out)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser("Arguments for data processor")
    cmd.add_argument('--config_path', type=str)
    cmd.add_argument('--corpus_path', type=str, required=True)
    cmd.add_argument('--vocab_dir', type=str)
    cmd.add_argument('--output_path', type=str, required=True)
    cmd.add_argument('--keep_span', type=bool, default=False)
    cmd.add_argument('--task_type', choices=['split', 'tokenizing', 'sampling'], default='split')
    args = cmd.parse_args(sys.argv[1:])

    # For splitting wiki corpus
    if args.task_type == 'split':
        split_wiki_corpus(args.corpus_path, args.output_path, 
                        r'(\s\.\s)', [" @-@ ", " @,@ ", " @.@ "])
    elif args.task_type == 'tokenizing':
        # For converting english text to ids
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(args.config_path)
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir, config=config, use_fast=True)
        if not args.keep_span:
            convert_txt_to_ids(args.corpus_path, tokenizer, args.output_path, max_len=200)
        else:
            convert_txt_to_ids_spans(args.corpus_path, tokenizer, )
    elif args.task_type == 'sampling':
        random_select_sentences(args.corpus_path, args.output_path, [50, 100, 200, 500], 1000)