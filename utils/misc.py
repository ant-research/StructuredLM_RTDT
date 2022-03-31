import unicodedata
import numpy as np


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


def _align_spans(original_spans, token_spans):
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

def load_vocab(vocab_file):
    word2idx = {}
    idx2word = {}
    for line in open(vocab_file, 'r', encoding='utf-8'):
        v, k = line.strip().split()
        word2idx[v] = int(k)
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return word2idx, idx2word