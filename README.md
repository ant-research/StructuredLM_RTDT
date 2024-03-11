# Composition Model

This library aims to construct syntactic compositional representations for text in an unsupervised manner. The covered areas may involve interpretability, text encoders, and generative language models.


## Milestones
"[R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling](https://aclanthology.org/2021.acl-long.379/)" (ACL2021), [R2D2](https://github.com/alipay/StructuredLM_RTDT/tree/r2d2)

Proposing an unsupervised syntactic language model of linear complexity, based on a neural inside algorithm with heuristic pruning.

"[Fast-R2D2: A Pretrained Recursive Neural Network based on Pruned CKY for Grammar Induction and Text Representation](https://arxiv.org/abs/2203.00281)". (EMNLP2022),[Fast_r2d2](https://github.com/alipay/StructuredLM_RTDT/tree/fast-R2D2)

Improve the heuristic pruning module in R2D2 to model-based pruning.


"[A Multi-Grained Self-Interpretable Symbolic-Neural Model For Single/Multi-Labeled Text Classification](https://openreview.net/forum?id=MLJ5TF5FtXH)".(ICLR 2023), [self-interpretable classification](https://github.com/ant-research/StructuredLM_RTDT/tree/self_interpretable_classification)

We explore the interpretability of the structured encoder and find that the induced alignment between labels and spans is highly consistent with human rationality.

"[Augmenting Transformers with Recursively Composed Multi-Grained Representations](https://openreview.net/forum?id=u859gX7ADC)". (ICLR 2024) current branch.

We reduce the space complexity of the deep inside-outside encoder from cubic to linear and further reduce the parallel time complexity to approximately log N. Meanwhile, we find that joint pre-training of Transformers and composition models can enhance a variety of NLP downstream tasks.

## Setup

Compile C++ codes.

python setup.py build_ext --inplace