cd trainer

torchrun --standalone --nnodes=1 --nproc_per_node=1 r2d2+_mlm_pretrain.py \
    --config_path ../data/en_config/r2d2+_config.json \
    --model_type cio --parser_lr 1e-3 \
    --corpus_path ../../corpus/wiki103.sent/wiki.train.split \
    --input_type bin --vocab_path ../data/en_config \
    --epochs 10 --output_dir ../test_pretrain \
    --min_len 2 --log_step 10 --batch_size 64 --max_batch_len 512 \
    --save_step 2000 --cache_dir ../test_pretrain_cache \
    --coeff_decline 0.00 --ascending 