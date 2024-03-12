    python experiments/span_train.py \
    -data_path data/ontonotes/srl \
    -config_path data/en_config/r2d2+_config.json \
    -pretrain_dir data/r2d2+_noshare_30 \
    -task src -model_type r2d2 \
    -mlm_rate 0.15 -decline_rate 0.015 \
    -fine_tune \
    -use_argmax \
    -criteria ce \
    -slurm_comment noshare