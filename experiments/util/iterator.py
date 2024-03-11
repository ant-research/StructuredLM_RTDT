from torch.utils.data.sampler import SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch
import random

pad_token_id = 0
class FixLengthLoader(object):
    def __init__(self, dataset, batch_size, shuffle, 
                 mask_id, mlm_rate=0.0, decline_rate=0.0, mask_epochs=10):
        self.dataset = dataset
        self.shuffle = shuffle
        # TODO: shuffle is not implement!
        self.bs = batch_size
        self.sampler = SequentialSampler(self.dataset)
        self.item_dict_key = ['subwords','subword_to_word_idx','spans1','spans2']
        self.item_lst_key = ['labels','seq_len']
        self._mask_id = mask_id
        self._mlm_rate = mlm_rate
        # self._decline_rate = decline_rate
        self._mask_epochs = mask_epochs
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        
    def _get_mask_rate(self):
        if self._epoch < self._mask_epochs:
            return self._mlm_rate
        else:
            return 0.0
        
    def shuffle_self(self):    
        if self.shuffle:
            self.dataset.reorder()

    def __iter__(self):
        batch = []
        length = 0
        for rec in self.dataset:
            length_getter = lambda x: self.dataset.instance_length_getter(x)
            if len(batch) > 0 and (len(batch) >= self.bs or length_getter(rec) != length):
            # explain above: if batch is not empty and 
            # (batch size is larger than batch size or length of current record is not equal to length of previous record)
            # why not just use len(batch) >= self.bs?
            # because the length of records in batch is not equal, so we need to pad them to the same length
                yield self.make_batch(batch)
                batch = []
            batch.append(rec)
            length = length_getter(rec)
        if len(batch) > 0:
            yield self.make_batch(batch)

    def __len__(self):
    # this method means the number of batches
        lm = self.dataset.length_map
        l = 0
        for i in lm:
            l += lm[i] // self.bs
            if lm[i] % self.bs > 0:
                l += 1
        return l

    def make_batch(self, batch):
        encoder_key_lst = list(batch[0]['subwords'].keys())
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = {} if key in self.item_dict_key else []
        if self._mlm_rate > 0:
            batch_dict['targets'] = {}
            batch_dict['parser_subwords'] = {}
        for rec in batch:
            for k in rec:
                if k in self.item_dict_key:
                    if rec[k] != None: # 'ctl', 'nel' dont have spans2  ('spans2': spans['span2'] if len(spans['span2']) > 0 else None in data.py)
                        for encoder_key in encoder_key_lst:
                            if encoder_key not in batch_dict[k]:
                                batch_dict[k][encoder_key] = []
                                
                            # MLM preparation 
                            if self._mlm_rate > 0 and k == 'subwords':
                                if encoder_key not in batch_dict['targets']:
                                    batch_dict['targets'][encoder_key] = []
                                    batch_dict['parser_subwords'][encoder_key] = []
                                # NOTE: need to create a copy to avoid accumulated masking
                                input_ids = rec[k][encoder_key].detach().clone() # subwords are already tensors
                                orig_ids = rec[k][encoder_key].detach().clone()
                                tgt = torch.zeros(input_ids.shape[0], dtype=torch.long).fill_(-1)
                                # mask_rate = max(0, self._mlm_rate - self._decline_rate * self._epoch)
                                mask_rate = self._get_mask_rate()
                                mask_len = round(input_ids.shape[0] * mask_rate)
                                if mask_len > 0:
                                    mask_ids = random.sample(range(input_ids.shape[0]), mask_len)
                                    tgt[mask_ids] = input_ids[mask_ids]
                                    input_ids[mask_ids] = self._mask_id  
                                batch_dict[k][encoder_key].append(input_ids)
                                # MLM-specific features
                                batch_dict['targets'][encoder_key].append(tgt)
                                batch_dict['parser_subwords'][encoder_key].append(orig_ids)
                                
                            else:    
                                batch_dict[k][encoder_key].append(rec[k][encoder_key])             
                else:
                    batch_dict[k].append(rec[k])
        for encoder_key in encoder_key_lst:
            batch_dict['subwords'][encoder_key] = pad_sequence(batch_dict['subwords'][encoder_key], batch_first=True, padding_value=pad_token_id).long()
            if self._mlm_rate > 0:
                batch_dict['targets'][encoder_key] = pad_sequence(batch_dict['targets'][encoder_key], batch_first=True, padding_value=-1).long()
                batch_dict['parser_subwords'][encoder_key] = pad_sequence(batch_dict['parser_subwords'][encoder_key], batch_first=True, padding_value=pad_token_id).long()
                if torch.cuda.is_available():
                    batch_dict['targets'][encoder_key] = batch_dict['targets'][encoder_key].cuda()
                    batch_dict['parser_subwords'][encoder_key] = batch_dict['parser_subwords'][encoder_key].cuda()            
            if torch.cuda.is_available():
                batch_dict['subwords'][encoder_key] = batch_dict['subwords'][encoder_key].cuda()         
        batch_dict['seq_len'] = torch.tensor(batch_dict['seq_len'])

        for encoder_key in encoder_key_lst:
            subword_to_word_idx_list = batch_dict['subword_to_word_idx'][encoder_key]
            subword_to_word_idx = pad_sequence(subword_to_word_idx_list, batch_first=True, padding_value=-1)
            batch_dict['subword_to_word_idx'][encoder_key] = subword_to_word_idx
        return batch_dict
