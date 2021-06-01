# R2D2 

This is the official code for paper titled "R2D2: Recursive Transformer based on Differentiable Tree for Interpretable Hierarchical Language Modeling".

## Data

## Train

### Multi-GPUs

For training from scratch in a single machine with multiple GPUs, please follow scripts below:

```bash
CORPUS_PATH=
OUTPUT_PATH=

python -m torch.distributed.launch \
    --nproc_per_node=$1 R2D2_trainer.py --batch_size 16 \
    --min_len 2 \
    --max_len 16 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 10 \
    --output_dir $OUTPUT_PATH \
    --window_size 4 \
    --input_type ids
```

### Single-GPU

```bash
CORPUS_PATH=
OUTPUT_PATH=

python -m trainer.R2D2_trainer \
    --batch_size 16 \
    --min_len 2 \
    --max_len 16 \
    --max_line -1 \
    --corpus_path $CORPUS_PATH \
    --vocab_path data/en_bert/bert-base-uncased-vocab.txt \
    --config_path data/en_bert/config.json \
    --epoch 10 \
    --output_dir $OUTPUT_PATH \
    --input_type txt
```

## Inference

```bash

```

## Evaluation

```bash

```
## Contact 

aaron.hx@alibaba-inc.com and haitao.mi@alibaba-inc.com