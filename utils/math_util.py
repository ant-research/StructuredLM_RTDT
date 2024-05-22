import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits):
    max = np.max(logits)
    logits = logits - max
    exp_x = np.exp(logits)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def max_neg_value(dtype):
    return -torch.finfo(dtype).max


def gumbel_softmax(logits, temperature=1, hard=True, train=False):
    """
    ST-gumple-softmax
    input: [*, seq_len, seq_len]
    return: flatten --> [*, seq_len, seq_len] an one-hot vector
    """
    if train:
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
    else:
        y = F.softmax(logits, dim=-1)
        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard