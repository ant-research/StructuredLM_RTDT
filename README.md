# Composition Model

This library aims to construct syntactic compositional representations for text in an unsupervised manner. The covered areas may involve interpretability, text encoders, and generative language models.


## Milestones
"[R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling](https://aclanthology.org/2021.acl-long.379/)" (ACL2021), [R2D2](https://github.com/alipay/StructuredLM_RTDT/tree/r2d2)

Proposing an unsupervised structured encoder able to compose low-level constituents into high-level constituents without gold trees. The learned trees are highly consistent with human-annotated ones. The backbone of the encoder is a neural inside algorithm with heuristic pruning, thus the time and space complexity are both in linear.

"[Fast-R2D2: A Pretrained Recursive Neural Network based on Pruned CKY for Grammar Induction and Text Representation](https://arxiv.org/abs/2203.00281)". (EMNLP2022),[Fast_r2d2](https://github.com/alipay/StructuredLM_RTDT/tree/fast-R2D2)

Improve the heuristic pruning module used in R2D2 to model-based pruning.


"[A Multi-Grained Self-Interpretable Symbolic-Neural Model For Single/Multi-Labeled Text Classification](https://openreview.net/forum?id=MLJ5TF5FtXH)".(ICLR 2023), [self-interpretable classification](https://github.com/ant-research/StructuredLM_RTDT/tree/self_interpretable_classification)

We explore the interpretability of the structured encoder and find that the induced alignment between labels and spans is highly consistent with human rationality.

"[Augmenting Transformers with Recursively Composed Multi-Grained Representations](https://openreview.net/forum?id=u859gX7ADC)". (ICLR 2024) [ReCAT](https://github.com/ant-research/StructuredLM_RTDT/tree/ReCAT)

We reduce the space complexity of the deep inside-outside algorithm from cubic to linear and further reduce the parallel time complexity to approximately log N thanks to the new pruning algorithm proposed in this paper. Furthermore, we find that joint pre-training of Transformers and composition models can enhance a variety of NLP downstream tasks.

"[Generative Pretrained Structured Transformers: Unsupervised Syntactic Language Models at Scale](http://arxiv.org/abs/2403.08293)". (ACL2024)  (current main branch)

We propose GPST, a syntactic language model which could be pre-trained on raw text efficiently without any human-annotated trees. When GPST and GPT-2 are both pre-trained on OpenWebText from scratch, GPST can outperform GPT-2 on various downstream tasks. Moreover, it significantly surpasses previous methods on generative grammar induction tasks, exhibiting a high degree of consistency with human syntax.


# README
The README will be gradually improved over the next two weeks.

## Contact

aaron.hx@antgroup.com