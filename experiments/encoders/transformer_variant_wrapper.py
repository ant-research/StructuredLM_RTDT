import os
import transformers
from transformers import AutoModel
from packaging import version

import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerVariantWrapper(nn.Module):
    def __init__(self, config, tokenizer, pretrain_dir) -> None:
        super().__init__()
        self.cls_id = config.cls_id
        self.model = AutoModel.from_config(config)
        self.tokenizer = tokenizer
        self.cls_dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.attention_probs_dropout_prob)
        )
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.hidden_size = config.hidden_size
        
        model_path = os.path.join(pretrain_dir, 'model30_1622.bin')
        self.from_pretrain(model_path, strict=False)
        
        # Set shift size due to introduction of special tokens
        self.start_shift = (1 if self.tokenizer._cls_token else 0)
        self.end_shift = (1 if self.tokenizer._sep_token else 0)
        
    def _tie_weights(self):
        self.classifier.weight = self.model.embeddings.word_embeddings.weight

    def from_pretrain(self, model_path, strict=True):
        state_dict = torch.load(model_path, map_location=lambda a, b: a)
        transfered_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            transfered_state_dict[new_k] = v
        self.load_state_dict(transfered_state_dict, strict=strict)
        self._tie_weights()
    
    def tokenize(self, sentence, get_subword_indices=False, force_split=False):
        """
        sentence: A single sentence where the sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: A list of length L, # of tokens, or a pair of L-length lists
            if get_subword_indices is set to True.
        """
        tokenizer = self.tokenizer
        subword_to_word_idx = []

        if not get_subword_indices:
            # Operate directly on a string
            if type(sentence) is list:
                sentence = ' '.join(sentence)
            token_ids = tokenizer.encode(sentence, add_special_tokens=True)
            return token_ids

        elif get_subword_indices:
            # Convert sentence to a list of words
            if type(sentence) is list:
                # If list then don't do anything
                pass
            elif force_split:
                sentence = sentence.strip().split()
            else:
                try:
                    sentence = tokenizer.basic_tokenizer.tokenize(sentence)
                except AttributeError:
                    # Basic tokenizer is not a part of Roberta
                    sentence = sentence.strip().split()

           
            token_ids = []
            for word_idx, word in enumerate(sentence):
                subword_list = tokenizer.tokenize(word)
                subword_ids = tokenizer.convert_tokens_to_ids(subword_list)

                subword_to_word_idx += [word_idx] * len(subword_ids)
                token_ids += subword_ids

            # Add special tokens
            if version.parse(transformers.__version__) > version.parse('2.0.0'):
                # Long term API
                final_token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
            else:
                # This API was in use when we started
                final_token_ids = tokenizer.add_special_tokens_single_sequence(token_ids)

            # Add -1 to denote the special symbols
            subword_to_word_idx = (
                [-1] * self.start_shift + subword_to_word_idx + [-1] * self.end_shift)
            
            return final_token_ids, subword_to_word_idx


    def forward(self, 
                input_ids,
                target_ids=None,
                add_cls=False,
                **kwargs):
        # add cls to input_ids
        masks = (input_ids != self.tokenizer.pad_token_id).cuda().float()
        padded_input_ids = torch.full((input_ids.shape[0], input_ids.shape[1] + 1), \
            dtype=torch.long, fill_value=self.cls_id, device=input_ids.device)
        padded_input_ids[:, 1:] = input_ids
        padded_mask = torch.ones(masks.shape[0], masks.shape[1] + 1, \
            dtype=torch.float, device=masks.device)
        padded_mask[:, 1:] = masks
        outputs = self.model(padded_input_ids, padded_mask)
        last_hidden = outputs.last_hidden_state
        
        logits = self.classifier(self.cls_dense(last_hidden))
        if target_ids is not None:
            lm_loss = F.cross_entropy(logits[:, 1:, :].permute(0, 2, 1), target_ids, ignore_index=-1)
        else:
            lm_loss = torch.zeros((1,), dtype=torch.float, device=input_ids.device)
            
        results = {}
        results['loss'] = lm_loss
        results['cls_embedding'] = last_hidden[:, 0, :]
        results['token_embeddings'] = last_hidden[:, 1:, :]
        
        return results['token_embeddings']