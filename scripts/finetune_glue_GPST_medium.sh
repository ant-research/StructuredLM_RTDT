# Take SST-2 as an example
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer/glue_trainer.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_512_4_1.json \
    --gpt_config_path data/gpt2-medium/config.json \
    --vocab_dir data/gpt2-medium \
    --glue_dir PATH_TO_SST-2_DATASET \
    --task_name GLUE_TASK_NAME(sst-2) \
    --output_dir FINETUNED_MODEL_SAVE_DIR \
    --pretrain_dir PATH_TO_PRETRAINED_MODEL \
    --log_step 20 \
    --save_step 500000 \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --parser_lr 1e-3 \
    --pool_size 3 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --coeff_proportion 1.0 \
    --gradient_checkpoint \
    --finetune_type discriminant \
    --eval_perepoch \