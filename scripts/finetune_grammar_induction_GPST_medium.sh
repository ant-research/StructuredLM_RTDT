export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer/text_finetune.py \
    --model_type r2d2-gen-fast \
    --eval_mode generative \
    --r2d2_config_path data/en_config/r2d2_512_4_1.json \
    --gpt_config_path data/gpt2-medium/config.json \
    --vocab_dir data/gpt2-medium \
    --corpus_path PATH_TO_WSJ_TRAINDATA_RAW_TEXT \
    --valid_corpus_path PATH_TO_WSJ_VALIDDATA_RAW_TEXT \
    --test_corpus_path PATH_TO_WSJ_TESTDATA_RAW_TEXT \
    --output_dir FINETUNED_CHECKPOINTS_SAVE_DIR \
    --pretrain_dir PATH_TO_PRETRAINED_MODEL \
    --batch_size 32 \
    --eval_batch_size 32 \
    --accumulation_steps 1 \
    --pool_size 1 \
    --epoch 10 \
    --log_steps 50 \
    --save_step 2000 