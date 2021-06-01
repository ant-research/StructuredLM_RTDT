# coding=utf-8
# Copyright (c) 2021 Ant Group

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


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, train):
    if train:
        y = logits + sample_gumbel(logits.size(), logits.device)
    else:
        y = logits
    return F.softmax(y / temperature, dim=-1), y


def hard_softmax(logits):
    y = F.softmax(logits, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def gumbel_softmax(logits, temperature=1, hard=True, train=False):
    """
    ST-gumple-softmax
    input: [*, seq_len, seq_len]
    return: flatten --> [*, seq_len, seq_len] an one-hot vector
    """
    y, _logits = gumbel_softmax_sample(logits, temperature, train)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard
