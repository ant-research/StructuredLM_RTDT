# R2D2/Fast-R2D2

This is the official code for paper titled "[R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling](https://arxiv.org/abs/2107.00967)".
The current repo is refactored from the original version used in the paper. If meet any issue, please feel free to feedback.

Our new work Fast-R2D2 will be released soon~ 

## Data

## Train

### Multi-GPUs

For training from scratch in a single machine with multiple GPUs, please follow scripts below:

```bash
CORPUS_PATH=
OUTPUT_PATH=
NODE_NUM=

python -m torch.distributed.launch \
    --nproc_per_node $NODE_NUM R2D2_trainer.py --batch_size 16 \
    --min_len 2 \
    --max_batch_len 512 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 60 \
    --output_dir $OUTPUT_PATH \
    --window_size 4 \
    --input_type txt
```

### Single-GPU

```bash
CORPUS_PATH=
OUTPUT_PATH=

python trainer.R2D2_trainer \
    --batch_size 16 \
    --min_len 2 \
    --max_batch_len 512 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 10 \
    --output_dir $OUTPUT_PATH \
    --input_type txt
```


## Evaluation

Evaluating the bidirectional language model task.
```bash
CORPUS_PATH=path to training corpus
VOCAB_DIR=directory of vocab.txt
MODEL_PATH=path to model.bin
CONFIG_PATH=path to config.json

python lm_eval_buckets.py \
    --model_name R2D2 \
    --dataset test \
    --config_path CONFIG_PATH \
    --model_path MODEL_PATH \
    --vocab_dir VOCAB_DIR \
    --corpus_path CORPUS_PATH
```

For evaluating F1 score on constituency trees, please refer to https://github.com/harvardnlp/compound-pcfg/blob/master/compare_trees.py

Evaluating compatibility with dependency trees:
Download WSJ dataset and convert to dependency trees by Stanford CoreNLP(https://stanfordnlp.github.io/CoreNLP/).
As WSJ is not a free dataset, it's not included in our project. Please refer to the files in data/predict_trees for detail format of tree induced.

```bash

python eval_tree.py \
    --pred_tree_path path_to_tree_induced \
    --ground_truth_path path_to_dependency_trees
    --vocab_dir VOCAB_DIR
```

## On-going work

1. Re-implement whole model to increase GPU utility ratio.
2. Pre-train on large corpus

## Contact 

aaron.hx@alibaba-inc.com and haitao.mi@alibaba-inc.com