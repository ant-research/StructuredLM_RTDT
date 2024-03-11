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

## Dataset preprocessing
Dataset: WikiText-103

Before pre-training, we preprocess corpus by spliting raw texts to sentences, tokenizing them, and converting them into numpy format.

Split texts into sentences

python utils/data\_processor.py --corput\_path PATH\_TO\_YOUR\_CORPUS --task\_type split --output\_path PATH\_TO\_SPLIT_CORPUS

Tokenize raw texts and convert them into numpy format.

python utils/dataset\_builder.py

## Pre-training
cd trainer

torchrun --standalone --nnodes=1 --nproc\_per\_node=1 r2d2+\_mlm\_pretrain.py 
    --config\_path ../data/en_config/r2d2+\_config.json 
    --model\_type cio --parser_lr 1e-3 
    --corpus\_path ../../corpus/PATH\_TO\_PREPROCESSED\_CORPUS 
    --input\_type bin --vocab\_path ../data/en\_config 
    --epochs 10 --output\_dir ../PRETRAIN\_MODEL\_SAVE\_DIR 
    --min\_len 2 --log\_step 10 --batch\_size 64 --max\_batch\_len 512 
    --save\_step 2000 --cache_dir ../pretrain\_cache 
    --coeff\_decline 0.00 --ascending 

## Downstreawm tasks

Please refer to scripts.

## Contact

aaron.hx@antgroup.com