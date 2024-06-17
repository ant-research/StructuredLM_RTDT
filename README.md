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


# Overview
Trees learned unsupervisedly

<img src="images/showcase1.png" width="700">
<img src="images/showcase2.png" width="500">

## Illustration of GPST generation process
Here is an illustration of the syntactic generation process for the sentence "fruit flies like a banana".
<img src="images/GPST_generation_process.gif" width="800">

## Illustration of how the neural inside pass works
<img src="images/composition_model.gif" width="640">

## Illustration of pruned neural inside pass
<img src="images/fast_inside_outside.gif" width="640">

## Illustration of parallel training of GPST
<img src="images/GPST_parallel_training.gif" width="640">


# README

## Setup

Compile C++ codes.

`python setup.py build_ext --inplace`

## Corpus preprocessing

Dataset: WikiText-103 and OpenWebText.

Before pre-training, we preprocess corpus by spliting raw texts to sentences, tokenizing them, and converting them into numpy memory-mapped format.

Raw text acquiring:

WikiText-103: https://huggingface.co/datasets/wikitext
Download link reference: https://developer.ibm.com/exchanges/data/all/wikitext-103/

OpenWebText: https://huggingface.co/datasets/Skylion007/openwebtext
Download link reference: https://zenodo.org/records/3834942

Raw text preprocessing: `sh scripts/preprocess_corpus.sh`

## Pre-training

To pretrain GPST<sub>medium</sub>: `sh scripts/pretrain_GPST_medium.sh` 

To pretrain GPST<sub>small</sub>: `sh scripts/pretrain_GPST_small.sh` 

## Downstream Tasks

### GLUE

#### Data Acquiring

GLUE: https://huggingface.co/datasets/nyu-mll/glue
Download link reference: https://gluebenchmark.com/tasks/ 

#### Scripts

To finetune GPST<sub>medium</sub> on GLUE: `sh scripts/finetune_glue_GPST_medium.sh` 

To finetune GPST<sub>small</sub> on GLUE: `sh scripts/finetune_glue_GPST_small.sh` 

### Summary Tasks

#### Data Acquiring and Preprocessing

We acquire datasets in parquet format from huggingface and do preprocessing on them.

XSum: https://huggingface.co/datasets/EdinburghNLP/xsum

CNN-DailyMail: https://huggingface.co/datasets/abisee/cnn_dailymail

Gigaword: https://huggingface.co/datasets/Harvard/gigaword

Summary dataset preprocessing: `sh scripts/preprocess_summary_dataset.sh`

#### Scripts

To finetune GPST<sub>medium</sub> on Summary Tasks: `sh scripts/finetune_summary_GPST_medium.sh` 

To evaluate finetuned GPST<sub>medium</sub> checkpoints: `sh scripts/evaluate_summary_GPST_medium.sh` 

To finetune GPST<sub>small</sub> on Summary Tasks: `sh scripts/finetune_summary_GPST_small.sh` 

To evaluate finetuned GPST<sub>small</sub> checkpoints: `sh scripts/evaluate_summary_GPST_small.sh` 

### Grammar Induction

#### Data Acquiring

WSJ: https://paperswithcode.com/dataset/penn-treebank
Download link reference: https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view

We further convert training data to raw text version.

#### Scripts

To finetune GPST<sub>medium</sub> on Grammar Induction: `sh scripts/finetune_grammar_induction_GPST_medium.sh` 

To evaluate finetuned GPST<sub>medium</sub> checkpoints: `sh scripts/evaluate_grammar_induction_GPST_medium.sh` 
then `sh scripts/compare_trees.sh`

To finetune GPST<sub>small</sub> on Grammar Induction: `sh scripts/finetune_grammar_induction_GPST_small.sh` 

To evaluate finetuned GPST<sub>small</sub> checkpoints: `sh scripts/evaluate_grammar_induction_GPST_small.sh` 
then `sh scripts/compare_trees.sh`

For evaluating F1 score on constituency trees, please refer to https://github.com/harvardnlp/compound-pcfg/blob/master/compare_trees.py

### Syntactic Generalization

#### Data Acquiring and Preprocessing

We acquire datasets in json format from github and do preprocessing on them.

Syntactic Generalization test suites: https://github.com/cpllab/syntactic-generalization/tree/master/test_suites/json

Syntactic Generalization test suites preprocessing: `sh scripts/preprocess_sg_dataset.sh`

#### Scripts

To evaluate GPST<sub>medium</sub>: `sh scripts/evaluate_sg_GPST_medium.sh`

To evaluate GPST<sub>small</sub>: `sh scripts/evaluate_sg_GPST_small.sh`

## Contact

aaron.hx@antgroup.com