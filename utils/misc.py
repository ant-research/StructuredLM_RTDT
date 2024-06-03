import unicodedata
import numpy as np
import math

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def hasattr(self, val):
        return val in self


def convert_char_span_to_tokenized_span_atis(offset_mappings, char_start, char_end):
    token_start_pos = token_end_pos = -1
    for token_pos, (char_st, char_ed) in enumerate(offset_mappings):
        if char_st == char_start:
            token_start_pos = token_pos
        if char_ed == char_end:
            token_end_pos = token_pos

    if token_start_pos == -1 or token_end_pos == -1:
        for token_pos, (char_st, char_ed) in enumerate(offset_mappings):
            if char_st == char_start -1:
                token_start_pos = token_pos
            if char_ed == char_end - 1:
                token_end_pos = token_pos
    assert token_start_pos != -1
    assert token_end_pos != -1
    return token_start_pos, token_end_pos


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]:
            continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids


def _find_point_in_spans(point, start_index, spans):
    """

        Find which subword token a point in the original sentence lies in.

    Args:
            point(int): an index in the sentence string
            start_index(int): the index of the subword in `spans`
                                to start searching
            spans(list[tuple[int]]): huggingface tokenizers' offset_mapping
                                        each subword's index span in the sentence string


    Returns:    
                index(int): the index of the subword in `spans`
                                that `point` belongs to

    """
    index = start_index
    while index < len(spans):
        span = spans[index]
        if span is not None and span[0] < span[1]:  # span is not empty
            if point >= span[0] and point < span[1]:
                break
        else:
            assert span is None or span[0] == span[1] == 0
        index += 1
    return index


def align_spans(original_spans, token_spans):
    """
    
    Map each word to its subtokens.

    Args:   original_spans(list[tuple[int]]): slice indices to index each word 
                                                in the sentence string (word_sep considered)
            token_spans(list[tuple[int]]): huggingface tokenizers' offset_mapping
                                                each subword's index span in the sentence string
    Returns:
            word_starts(list[int]): word_starts[i]=j means ith word begins at jth subword
            word_ends(list[int]): word_ends[i]=j means ith word ends in jth subword (inclusive)
    """
    word_starts = []
    word_ends = []

    while token_spans and (token_spans[-1] is None or token_spans[-1][1] == 0):
        token_spans.pop()  # remove trailing empty spans

    last = 0
    for (start, end) in original_spans:
        first = _find_point_in_spans(start, last, token_spans)
        last = _find_point_in_spans(end - 1, first, token_spans)

        word_starts.append(first)
        word_ends.append(last)

    return word_starts, word_ends


def get_sentence_from_words(words, word_sep):
    """

    Join words by word_sep and record each word's boundary.

    Args：
            words(list[str]): each element is a word
            word_sep(str): str used to join the words, default: ' '
    Returns:
            sentence(str): words joined by word_sep
            word_char_spans(list[tuple[int]]): each word's index span in the sentence string 
                                                (word_sep considered)
    """
    sentence = []
    word_char_spans = []
    offset = 0

    for word in words:
        length = len(word)

        sentence.append(word)
        word_char_spans.append((offset, offset + length))

        offset += length + len(word_sep)

    sentence = word_sep.join(sentence)

    return sentence, word_char_spans

def padding(arr_list, pad_val=0):
    max_len = max([len(arr) for arr in arr_list])
    for arr in arr_list:
        arr.extend([pad_val] * (max_len - len(arr)))
    return arr_list

def gpt_token(gpt_token):
    return gpt_token.replace('Ġ', '')

def convert_token(gpt_token):
    return gpt_token.replace('Ġ', ' ')