import torch
import torch.nn as nn
import logging
from packaging import version

import transformers


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


class SimpleEncoder(nn.Module):
    def __init__(self, tokenizer, **kwargs):
        super(SimpleEncoder, self).__init__()
        
        self.tokenizer = tokenizer
        
        # Set shift size due to introduction of special tokens
        # NOTE: depends on the pretraining process
        self.start_shift = (1 if self.tokenizer._cls_token else 0)
        self.end_shift = (1 if self.tokenizer._sep_token else 0)

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

