import torch


def instance_f1_info(label, preds):
    label_ones = label.sum().item()  # true positive + false negative
    preds_ones = preds.sum().item()  # true positive + false positive
    correct_ones = (label * preds).sum().item() # true positive
    return correct_ones, preds_ones, label_ones


def f1_score(numerator, denom_p, denom_r):
    if numerator == 0:
        return 0
    p = float(numerator) / denom_p
    r = float(numerator) / denom_r
    return 2 * p * r / (p + r)


def print_example(encoder, label_itos, idx_text, span, label, pred, prob=None):
    text = str(len(idx_text))
    text += ": "
    for i in range(len(idx_text)):
        if i == span[0]:
            text += '{'
        text += encoder.tokenizer.convert_ids_to_tokens(int(idx_text[i]))
        if i == span[1]:
            text += '}'
        text += ' '
    text += 'correct_answer: '
    text += label_itos[label]
    text += '/pred_answer: '
    if pred >= 0:
        text += label_itos[pred]
    else:
        text += 'NO_Answer'
    print(text)
    if not (prob is None):
        text = ''
        TOPK = 5
        prob = torch.squeeze(prob)
        vals, idxs = torch.topk(prob, TOPK)
        idxs = idxs.tolist()
        for i in range(TOPK):
            text += label_itos[idxs[i]]
            text += ": "
            text += str(round(float(vals[i]), 2))
            text += "; "
    print(text)
