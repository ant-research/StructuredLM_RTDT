# R2D2 

This is the official code for paper titled "R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling". The model code will be released soon.

## Data

## Train

### Multi-GPUs

For training from scratch in a single machine with multiple GPUs, please follow scripts below:

```bash
CORPUS_PATH=

python -m torch.distributed.launch \
    --nproc_per_node=$1 R2D2_trainer.py --batch_size 16 \
    --min_len 2 \
    --max_len 16 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 10 \
    --window_size 4 \
    --input_type ids
```

### Single-GPU

```bash
CORPUS_PATH=

python -m trainer.R2D2_trainer \
    --batch_size 16 \
    --min_len 2 \
    --max_len 16 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 10 \
    --output_dir transformer_models/mwoz22 \
    --input_type txt
```

## Inference

```bash

```

## Evaluation

```bash

```

test access settings
