# Milestones

This library aims to construct syntactic compositional representations for text in an unsupervised manner. The covered areas may involve interpretability, text encoders, and generative language models.


## Milestones
"[R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling](https://aclanthology.org/2021.acl-long.379/)" (ACL2021 Oral), [R2D2_tag](https://github.com/alipay/StructuredLM_RTDT/tree/r2d2)

Proposing an unsupervised syntactic language model of linear complexity, based on a neural inside algorithm with heuristic pruning.

"[Fast-R2D2: A Pretrained Recursive Neural Network based on Pruned CKY for Grammar Induction and Text Representation](https://arxiv.org/abs/2203.00281)". (EMNLP2022),[Fast_r2d2_tag](https://github.com/alipay/StructuredLM_RTDT/tree/fast-R2D2)

Improve the heuristic pruning module in R2D2 to model-based pruning.


"[A Multi-Grained Self-Interpretable Symbolic-Neural Model For Single/Multi-Labeled Text Classification](https://openreview.net/forum?id=MLJ5TF5FtXH)".(ICLR 2023) current main branch

We explore the interpretability of the structured encoder and find that the induced alignment between labels and spans is highly consistent with human rationality.

"[Augmenting Transformers with Recursively Composed Multi-Grained Representations](https://openreview.net/forum?id=u859gX7ADC)". (ICLR 2024) code will be released soon.
We reduce the space complexity of the deep inside-outside encoder from cubic to linear, and in a parallel environment, we reduce the time complexity to approximately log N. Meanwhile, we find that joint pre-training of Transformers and a composition-based encoder can enhance a variety of NLP downstream tasks.

If you find our work helpful, please give us a star~

## Requires
gcc >= 5.0,
pytorch == 1.9.0+cu111,
cuda == 11.1

For other versions of pytorch, please make sure the corresponding version of CUDA has been installed.

## Setup

export PATH="/usr/local/gcc-version/bin:$PATH"
export CXX=g++

python setup.py build_ext --inplace

Check if r2d2lib is correctly compiled:
python -m unittest unittests/cuda_unittest.py

## Dataset
WikiText103: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

GLUE: https://gluebenchmark.com/tasks

SNIPS: https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines

ATIS: https://github.com/howl-anderson/ATIS_dataset/tree/master/data

stanfordLU: https://atmahou.github.io/attachments/StanfordLU.zip

MIT SLS movie: https://groups.csail.mit.edu/sls/downloads/movie/

MIT SLS restaurant: https://groups.csail.mit.edu/sls/downloads/restaurant/

## Dataset preprocess

Split WikiText103 to sentences and remove special tokens like @-@, @.@.
```bash
CORPUS_PATH=
OUTPUT_PATH=

python -m utils.data_processor --corpus_path CORPUS_PATH --output_path OUTPUT_PATH --task_type split
```

Convert raw text to ids without span
```bash
CORPUS_PATH=path to your corpus
OUTPUT_PATH=path to processed corpus
CONFIG_PATH=
VOCAB_DIR=

python -m utils.data_processor --corpus_path $CORPUS_PATH --output_path $OUTPUT_PATH --task_type tokenizing \
    --vocab_dir $VOCAB_DIR --config_path $CONFIG_PATH
```

Convert raw text to ids with span
```bash
CORPUS_PATH=path to your corpus
OUTPUT_PATH=path to processed corpus
CONFIG_PATH=
VOCAB_DIR=

python -m utils.data_processor --corpus_path $CORPUS_PATH --output_path $OUTPUT_PATH --task_type tokenizing \
    --vocab_dir $VOCAB_DIR --config_path $CONFIG_PATH --keep_span
```

## Train
Pretrain. Whether including span constrains depends on whether convert to ids with span.
```bash
VOCAB_DIR=data/en_config
CONFIG_PATH=data/en_config/fast_r2d2.json
PROCESSED_CORPUS=output corpus process at the last step(tokenized and converted to ids)
OUTPUT_DIR=output model dir

cd trainer

python -m torch.distributed.launch --nproc_per_node 8 fast_r2d2_pretrain.py \
    --batch_size 96 --max_batch_len 1536 \
    --lr 5e-5 --parser_lr 1e-2 \
    --vocab_dir $VOCAB_DIR \
    --config_path $CONFIG_PATH \
    --max_grad_norm 1.0 --input_type ids \
    --corpus_path $PROCESSED_CORPUS \
    --output_dir $OUTPUT_DIR \
    --num_samples 256 --log_step 500 --epochs 60 \
    --seperator " "
```

## Grammar Induction

```bash
# generate trees in ptb format

python -m eval.r2d2_ptb_printer \
    --model_path path_to_r2d2_model \
    --parser_path path_to_r2d2_parser \
    --parser_only --in_word \
    --config_path \
    path_to_your_config \
    --corpus_path \
    data/wsj/wsj_test_raw.txt \
    --output_path \
    path_to_output_file

```

For evaluating F1 score on constituency trees, please refer to https://github.com/harvardnlp/compound-pcfg/blob/master/compare_trees.py

```bash
R2D2_TREE=path to output file generated by r2d2_ptb_printer

python compare_trees.py --tree1 path_to_gold_tree --tree2 R2D2_TREE
```


## GLUE tasks

finetune GLUE on 8*A100

```bash
TASK_NAME=SST-2/CoLA/QQP/MNLI

python -m torch.distributed.launch --nproc_per_node 8 trainer/fast_r2d2_glue_trainer.py \
    --max_grad_norm 1.0 --lr 5e-5 --parser_lr 1e-2 \
    --config_path path_to_config \
    --vocab_dir path_to_vocab_dir \
    --task_type sst-2 --glue_dir path_to_glue_dir/$TASK_NAME --max_batch_len 1536 \
    --max_batch_size 8 --output_dir path_to_model_save_dir \
    --epochs 10 --pretrain_dir path_to_pretrain_model_dir \
    --log_step 50 --num_samples 256 -sampler random --apex_mode O0
```

evaluation

```bash

TASK_TYPE=sst-2/mnli/cola/qqp
EVAL_TURN=number of the turn to evaluate
MODE= forced or cky

python -m eval.eval_fast_r2d2 \
    --model_dir \
    path_to_dir_of_fine_tuned_models \
    --config_path \
    path_to_config \
    --vocab_dir \
    dir_to_vocab \
    --task_type \
    TASK_TYPE \
    --glue_dir \
    dir_of_glue_task \
    --max_batch_len \
    1024 \
    --max_batch_size \
    32 \
    --turn \
    $EVAL_TURN \
    --r2d2_mode \
    $MODE
```

## Evaluate speed

Sample sentences from WikiText103(tokenized and converted to ids).

python -m utils.data_processor --task_type sampling --corpus_path path_to_wiki103_ids --output_path path_to_wiki103_outputs

```bash
LEN_RANGE=50/100/200/500
R2D2_MODE=forced/cky
BATCH_SIZE=50

python eval/eval_speed.py \
    --model_dir \
    path_to_pretrain_model_dir \
    --config_path \
    path_to_pretrain_model_dir/config.json \
    --vocab_dir \
    path_to_pretrain_model_dir \
    --corpus_path \
    model_data/en_wiki/wiki103.speed.ids.$LEN_RANGE \
    --max_batch_len \
    2500000 \
    --input_type \
    ids \
    --model \
    fast-r2d2 \
    --turn \
    59 \
    --r2d2_mode \
    $R2D2_MODE \
    --batch_size \
    $BATCH_SIZE
```

## Run experiments about self-interpretable classification model
The backbone for the self-interpretable classification model is in mode/fast_r2d2_dp_classification.

The scripts to run experiments described in our paper could be found under the scripts folder.

The pretrained Fast-R2D2 coud be found at [release](https://github.com/alipay/StructuredLM_RTDT/releases/tag/fast-R2D2)

## Contact 

aaron.hx@alibaba-inc.com
