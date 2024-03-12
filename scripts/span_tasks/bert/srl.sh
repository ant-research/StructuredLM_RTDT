    python experiments/span_train.py \
    -data_path data/ontonotes/srl \
    -task src -model_type bert \
    -criteria ce \
    -pool_methods max \
    -fine_tune \
    -use_argmax \
    -slurm_comment bert