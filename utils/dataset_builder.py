import os
import numpy as np
import pickle
import nltk
import codecs
import tarfile
from itertools import accumulate
import re


def build_dataset(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line]
                for sent in sents:
                    ids = tokenizer.encode(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)


def build_dataset_from_dir(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False):
    # filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    # with open(text_path, mode='r') as f_in:
    processed_files = 0
    for root, dirs, files in os.walk(texts_dir):
        for text_path in files:
            if processed_files % 10 == 0:
                print(f'processed: {processed_files} / {len(files)}', flush=True)
            processed_files += 1
            if text_path.endswith('_data'):
                tar = tarfile.open(os.path.join(root, text_path))
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    lines = file.readlines()
                    for line in lines:
                        line = line.decode().strip()
                        if len(line) == 0: # document split
                            if len(current_buffer) > 0:
                                flush(current_buffer)
                                current_buffer = []
                                item_lens.append(0)
                                # doc_num += 1
                                # if doc_num > 10:
                                #     break
                        else:
                            # tokenize to ids
                            if tokenize_sent:
                                sents = nltk.sent_tokenize(line)
                            else:
                                sents = [line]
                            for sent in sents:
                                ids = tokenizer.encode(sent)
                                current_buffer.append(ids)
                            if max_len > 0 and len(ids) >= max_len:
                                #drop
                                continue
                        
                    if len(current_buffer) > 0:
                        flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_dataset_batch(text_path_pattern, tokenizer, output_dir, buffer_size = 1024, max_len=200):
    import glob
    files = glob.glob(text_path_pattern)
    for f in files:
        pass

def print_dataset(index_path, data_path, tokenizer):
    np_memmap = np.memmap(data_path, dtype=np.int32, mode='r', order='C')
    with open(index_path, 'rb') as handle:
        item_lens = pickle.load(handle)
    
    ends = list(accumulate(item_lens))
    prev_end = 0
    for end in ends:
        print(tokenizer.convert_ids_to_tokens(np_memmap[prev_end : end]))
        prev_end = end
