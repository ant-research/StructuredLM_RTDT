import torch
import torch.nn as nn
import logging
from packaging import version

import transformers
from transformers import BertModel
from transformers import BertTokenizer


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Constants
MODEL_LIST = ['bert']
BERT_MODEL_SIZES = ['base', 'large']


class Encoder(nn.Module):
    def __init__(self, model='bert', model_size='base', cased=True,
                 fine_tune=False, **kwargs):
        super(Encoder, self).__init__()
        assert(model in MODEL_LIST)

        self.base_name = model
        self.model = None
        self.tokenizer = None
        self.num_layers = None
        self.hidden_size = None
        self.fine_tune = fine_tune

        # First initialize the model and tokenizer
        model_name = ''

        # Do we want the tokenizer to lower case or not
        do_lower_case = False
        if model == 'bert' and (not cased):
            # For other models this choice doesn't make sense since they are trained
            # on cased version of text.
            do_lower_case = True

        # Model is one of the BERT variants
        if 'bert' in model:
            assert (model_size in BERT_MODEL_SIZES)
            model_name = model + "-" + model_size
            if model == 'bert' and (not cased):
                # Only original BERT supports uncased models
                model_name += '-uncased'
            else:
                model_name += '-cased'

            model_path = model_name

            if model == 'bert':
                self.model = BertModel.from_pretrained(
                    model_path, output_hidden_states=True)
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_path, do_lower_case=do_lower_case)

            self.num_layers = self.model.config.num_hidden_layers + 1
            self.hidden_size = self.model.config.hidden_size

        # Set the model name
        self.model_name = model_name

        # Set shift size due to introduction of special tokens
        # NOTE: depends on the pretraining process
        self.start_shift = (1 if self.tokenizer._cls_token else 0)
        self.end_shift = (1 if self.tokenizer._sep_token else 0)

        # Set requires_grad to False if not fine tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        # Set parameters required on top of pre-trained models
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

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

            if self.base_name in ['bert']:
                token_ids = []
                for word_idx, word in enumerate(sentence):
                    subword_list = tokenizer.tokenize(word)
                    subword_ids = tokenizer.convert_tokens_to_ids(subword_list)

                    subword_to_word_idx += [word_idx] * len(subword_ids)
                    token_ids += subword_ids
            else:
                raise Exception("%s doesn't support getting word indices"
                                % self.base_name)

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

    def tokenize_sentence(self, sentence, get_subword_indices=False, force_split=False):
        """
        sentence: A single sentence where the sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: A tensor of size (1 x L) or a pair of (1 x L) tensors if get_subword_indices.
        """
        output = self.tokenize(
            sentence, get_subword_indices=get_subword_indices,
            force_split=force_split
        )
        if get_subword_indices:
            return (torch.tensor(output[0]).unsqueeze(dim=0).cuda(),
                    torch.tensor(output[1]).unsqueeze(dim=0).cuda())
        else:
            return torch.tensor(output).unsqueeze(dim=0).cuda()

    def tokenize_batch(self, list_of_sentences, get_subword_indices=False, force_split=False):
        """
        list_of_sentences: List of sentences where each sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: Padded tensors of size (B x L) or a pair of (B x L) tensors if get_subword_indices.
        """
        all_token_ids = []
        all_subword_to_word_idx = []

        sentence_len_list = []
        max_sentence_len = 0
        for sentence in list_of_sentences:
            if get_subword_indices:
                token_ids, subword_to_word_idx = \
                    self.tokenize(
                        sentence, get_subword_indices=True,
                        force_split=force_split
                    )
                all_subword_to_word_idx.append(subword_to_word_idx)
            else:
                token_ids = self.tokenize(sentence)

            all_token_ids.append(token_ids)
            sentence_len_list.append(len(token_ids))
            if max_sentence_len < sentence_len_list[-1]:
                max_sentence_len = sentence_len_list[-1]

        # Pad the sentences to max length
        all_token_ids = [
            (token_ids + (max_sentence_len - len(token_ids)) * [self.tokenizer.pad_token_id])
            for token_ids in all_token_ids
        ]

        if get_subword_indices:
            all_subword_to_word_idx = [
                (word_indices + (max_sentence_len - len(word_indices)) * [-1])
                for word_indices in all_subword_to_word_idx
            ]

        # Tensorize the list
        batch_token_ids = torch.tensor(all_token_ids).cuda()
        batch_lens = torch.tensor(sentence_len_list).cuda()
        if get_subword_indices:
            return (batch_token_ids, batch_lens,
                    torch.tensor(all_subword_to_word_idx))
        else:
            return (batch_token_ids, batch_lens)

    def forward(self, batch_ids, just_last_layer=False):
        """
        Encode a batch of token IDs.
        batch_ids: B x L
        just_last_layer: If True return the last layer else return a (learned) wtd avg of layers.
        """
        input_mask = (batch_ids != self.tokenizer.pad_token_id).cuda().float()

        if not self.fine_tune:
            with torch.no_grad():
                output_bert = self.model(
                    batch_ids, attention_mask=input_mask, output_hidden_states=True)  # B x L x E
                last_hidden_state = output_bert.last_hidden_state
                encoded_layers = output_bert.hidden_states
        else:
            output_bert = self.model(
                batch_ids, attention_mask=input_mask, output_hidden_states=True)  # B x L x E
            last_hidden_state = output_bert.last_hidden_state
            encoded_layers = output_bert.hidden_states
        # Encoded layers also has the embedding layer - 0th entry

        if just_last_layer:
            output = last_hidden_state
        else:
            wtd_encoded_repr = 0
            soft_weight = nn.functional.softmax(self.weighing_params, dim=0)

            for i in range(self.num_layers):
                wtd_encoded_repr += soft_weight[i] * encoded_layers[i]

            output = wtd_encoded_repr

        return output