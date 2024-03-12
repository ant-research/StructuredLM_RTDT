    python experiments/span_train.py \
    -data_path data/ontonotes/ner \
    -config_path data/r2d2+_noshare_30/config.json \
    -pretrain_dir data/r2d2+_noshare_30 \
    -task nel -model_type r2d2 \
    -mlm_rate 0.15 -decline_rate 0.015 \
    -fine_tune \
    -use_argmax \
    -criteria ce \
    -slurm_comment noshare