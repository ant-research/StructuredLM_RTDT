export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer/ddp_trainer_nosync.py \
    --r2d2_config_path data/en_config/r2d2_256_4_1.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --parser_lr 1e-3 \
    --lr 5e-5 \
    --corpus_path PATH_TO_PREPROCESSED_CORPUS_DIR \
    --output_dir PRETRAIN_MODEL_SAVE_DIR  \
    --batch_size 32 \
    --accumulation_steps 1 \
    --model_type r2d2-gen-fast \
    --num_samples 5000000_FOR_WIKI103_OR_15000000_FOR_OPENWEBTEXT \
    --gradient_checkpoint \
    --log_step 50 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --temperature_end 1.0 \
    --pool_size 1 \
    --save_step 10000