    python experiments/span_train.py \
    -data_path data/ontonotes/const/nonterminal \
    -config_path data/en_config/r2d2+_config.json \
    -pretrain_dir data/r2d2+_share_30 \
    -task ctl -model_type r2d2 \
    -share \
    -mlm_rate 0.15 -decline_rate 0.015 \
    -fine_tune \
    -criteria bce \
    -slurm_comment share