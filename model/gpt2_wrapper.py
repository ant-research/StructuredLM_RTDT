import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.model_loader import load_model
from model.gpt2_flash_attn import GPT2LMHeadModel
from model.modeling_outputs import R2D2GenOutput
from datetime import datetime
from filelock import FileLock
import os


class GPT2Wrapper(nn.Module):
    def __init__(self, config):
        # embedding dim is used to feed to r2d2
        # input dim is sued to feed to GPT
        super().__init__()
        self.gpt = GPT2LMHeadModel(config)
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id
        self.embedding_dim = config.n_embd
        
    def from_pretrain(self, model_path, **kwargs):
        self.gpt = self.gpt.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

    def forward(self, chunk_input_ids= None, chunk_masks=None, input_ids=None, masks=None, group_ids=None, 
                max_input_len=0, atom_spans=None, enable_gpt=True, coeff=1.0, eos_labels=None, **kwargs):
        if eos_labels is None:
            seq_lens = (chunk_masks != 0).sum(dim=1)
            tgt_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 1), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device).fill_(-100)
            # gpt_input_ids.fill_(-100)
            tgt_ids[:, 1:] = chunk_input_ids
            tgt_ids[:, 0] = self.bos_id
            gpt_input_ids = torch.where(tgt_ids != -100, tgt_ids, 0)
            result = self.gpt(input_ids=gpt_input_ids, labels=tgt_ids, return_dict=True)
            return R2D2GenOutput(non_struct_loss=result.loss, struct_loss = 0, logits=result.logits,
                                 tgt_ids=tgt_ids, splits=None)
        else:
            seq_lens = (chunk_masks != 0).sum(dim=1)
            tgt_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 2), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device)
            tgt_ids.fill_(-100)
            # gpt_input_ids.fill_(-100)
            tgt_ids[:, 1:-1] = chunk_input_ids
            tgt_ids[:, 0] = self.bos_id
            # set eos_label at the end of inputs
            tgt_ids.scatter_(1, seq_lens.unsqueeze(1), torch.tensor(eos_labels, device=chunk_input_ids.device).unsqueeze(1))
            gpt_input_ids = torch.where(tgt_ids != -100, tgt_ids, 0)
            # print("gpt_iunput_ids: ", gpt_input_ids, "tgt_ids: ", tgt_ids)
            result = self.gpt(input_ids=gpt_input_ids, labels=tgt_ids, return_dict=True)
            return R2D2GenOutput(non_struct_loss=result.loss, struct_loss = 0, logits=result.logits, splits=None)
