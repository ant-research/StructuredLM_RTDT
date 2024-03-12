cd trainer

torchrun --standalone --nnodes=1 --nproc_per_node=1 r2d2+_mlm_pretrain.py \
    --config_path ../data/en_config/r2d2+_config.json \
    --model_type cio --parser_lr 1e-3 \
    --corpus_path ../data/wsj/cpcfg_train_raw_keepspan.ids \
    --input_type ids --vocab_path ../data/en_config \
    --epochs 10 --output_dir ../OUTPUT_DIR \
    --min_len 2 --log_step 10 --batch_size 512 --max_batch_len 3072 \
    --save_step 2000 --cache_dir ../wsjkeepspan_batch_512_maxlen_3072_ascending \
    --coeff_decline 0.00 --ascending 