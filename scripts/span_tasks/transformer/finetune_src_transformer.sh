    python experiments/span_train.py \
    -data_path data/ontonotes/srl \
    -config_path data/en_config/transformer${1}_config.json \
    -pretrain_dir data/transformer${1}_30 \
    -task src -model_type transformer \
    -batch_size 64 \
    -eval_batch_size 4 \
    -criteria ce \
    -pool_methods max \
    -fine_tune \
    -use_argmax \
    -slurm_comment maxt$1