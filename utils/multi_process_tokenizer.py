import argparse
from math import ceil
import sys
import os
from transformers import AutoConfig, AutoTokenizer
from utils.data_processor import convert_txt_to_ids_spans
import logging
import codecs


def tokenize(proc_id, files, output_dir, config_dir, max_len=200):
    config = AutoConfig.from_pretrained(os.path.join(config_dir, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(config_dir, config=config, use_fast=True)
    for file_i, file in enumerate(files):
        convert_txt_to_ids_spans(file, tokenizer, seperator=' ', 
                                 output_path=os.path.join(output_dir, f'ids_{proc_id}_{file_i}.txt'),
                                 max_len=max_len)

def merge_files(files, output_file):
    with codecs.open(output_file, mode='w', encoding='utf-8') as f_out:
        for file in files:
            with codecs.open(file, mode='r', encoding='utf-8') as f_in:
                for line in f_in:
                    if len(line.strip()) > 0:
                        print(line, file=f_out)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser("Arguments for multi processor tokenizer")
    cmd.add_argument('--corpus_dir', type=str, required=True)
    cmd.add_argument('--output_dir', type=str, required=True)
    cmd.add_argument('--config_dir', type=str, required=True)
    cmd.add_argument('--proc_rate', default=0.75)
    cmd.add_argument('--task_type', choices=['tokenize', 'merge'], default='tokenize')
    args = cmd.parse_args(sys.argv[1:])

    logging.root.setLevel(logging.INFO)
    num_process = int(os.cpu_count() * args.proc_rate)

    file_list = []
    
    for root, dir, files in os.walk(args.corpus_dir):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file))
            if len(file_list) % 100 == 0:
                logging.info(f'{len(file_list)} added...')

    logging.info(f'total file size: {len(file_list)}, total_processor: {num_process}')

    if args.task_type == 'tokenize':
        chunk_size = ceil(len(file_list) / num_process)
        assert num_process * chunk_size >= len(file_list)

        from multiprocessing import Process
        processes = []
        for p_id in range(num_process):
            processes.append(Process(target=tokenize, args=(p_id, file_list[p_id * chunk_size: (p_id + 1) * chunk_size],
                                                            args.output_dir, args.config_dir)))
        
        [p.start() for p in processes]
        [p.join() for p in processes]
    elif args.task_type == 'merge':
        merge_files(file_list, os.path.join(args.output_dir, 'merged.txt'))
