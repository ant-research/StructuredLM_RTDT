import os
import numpy as np
import pickle
from itertools import accumulate


def build_dataset(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=200, 
                  compact=False):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'{filename}.idx')
    content_path = os.path.join(output_dir, f'{filename}.data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    sep_id = tokenizer.vocab['[SEP]']
    current_buffer = []
    
    def flush(buffer, split_doc=False):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        nd_arr = np.array(buffer, dtype=np.int32, order='C')
        while current_offset + len(buffer) > current_size:
            # expand
            np_memmap.flush()
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(current_size + buffer_size))
            current_size += buffer_size
        np_memmap[current_offset: current_offset + len(buffer)] = nd_arr
        current_offset += len(buffer)
        item_lens.append(len(buffer))
        if split_doc:
            item_lens.append(0)
        
        return np_memmap, current_offset
    
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            if len(line.strip()) == 0:
                if len(current_buffer) > 0:
                    flush(current_buffer, True)
                    current_buffer = []
            else:
                # tokenize to ids
                ids = tokenizer.encode(line)[1:-1]
                if len(ids) >= max_len:
                    #drop
                    continue
                    
                if compact and len(current_buffer) + len(ids) + 1 < max_len:
                    if len(current_buffer) > 0:
                        current_buffer.extend([sep_id] + ids)
                    else:
                        current_buffer.extend(ids)
                else:
                    if len(current_buffer) > 0:
                        flush(current_buffer)
                    current_buffer = ids
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)


def print_dataset(index_path, data_path, tokenizer):
    np_memmap = np.memmap(data_path, dtype=np.int32, mode='r', order='C')
    with open(index_path, 'rb') as handle:
        item_lens = pickle.load(handle)
    
    ends = list(accumulate(item_lens))
    prev_end = 0
    for end in ends:
        print(tokenizer.convert_ids_to_tokens(np_memmap[prev_end : end]))
        prev_end = end

        
if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("data/en_config")
    build_dataset("data/wsj/cpcfg_train_raw.txt", tokenizer, "data/processed_doc", 
                  buffer_size=16384, max_len=200, compact=False)
    