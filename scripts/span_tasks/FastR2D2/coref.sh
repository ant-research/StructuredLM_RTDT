    python experiments/span_train.py \
    -data_path data/ontonotes/coref \
    -config_path data/fast_r2d2/config.json \
    -pretrain_dir data/fast_r2d2 \
    -vocab_dir data/fast_r2d2 \
    -task coref -model_type fastr2d2 \
    -fine_tune \
    -use_argmax \
    -criteria ce \
    -slurm_comment fastr2d2 
