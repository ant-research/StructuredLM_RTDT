# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import abc
from typing import List


class TreeDecoderWrapper(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, tokens: List[str]):
        pass

    @abc.abstractmethod
    def print_binary_ptb(self, tokens) -> str:
        pass