    torchrun --standalone --nnodes=1 --nproc_per_node=$1 trainer/r2d2+_glue_trainer.py \
    --max_grad_norm 1.0 --lr 5e-5 --parser_lr 1e-2 \
    --config_path data/en_config/r2d2+_config.json \
    --vocab_dir data/en_config \
    --task_type mnli --glue_dir /GLUE_DIR/glue/MNLI \
    --cache_dir /CACHE_DIR/mnli/batch_256_maxlen_4000 \
    --max_batch_len 4000 --max_batch_size 256 \
    --output_dir  \
    --epochs 20  \
    --log_step 50 --eval_step 200 \
    --model_name fastr2d2+_iter \
    --enable_epoch_eval \
    --pretrain_dir /PATH_TO_YOUR_PRETRAIN_DIR/model.bin