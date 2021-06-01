# coding=utf-8
# Copyright (c) 2021 Ant Group

import abc

from typing import List

from data_structure.syntax_tree import BinaryNode


class TreeDecoderWrapper(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, tokens: List[str]) -> (BinaryNode, List[int]):
        pass

    @abc.abstractmethod
    def print_binary_ptb(self, tokens) -> str:
        pass