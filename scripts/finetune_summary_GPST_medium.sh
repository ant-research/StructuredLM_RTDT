# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer/xsum_trainer.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_512_4_1.json \
    --gpt_config_path data/gpt2-medium/config.json \
    --vocab_dir data/gpt2-medium \
    --summary_dir PATH_TO_GIGAWORD_DATASET \
    --output_dir FINETUNED_CHECKPOINTS_SAVE_DIR \
    --pretrain_dir PATH_TO_PRETRAINED_MODEL \
    --log_step 100 \
    --save_step 1000000000 \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --parser_lr 1e-3 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --pool_size 3 \
    --word_sync \
    --gradient_checkpoint \
    --document_threshold 400 \
    --summary_threshold 120
