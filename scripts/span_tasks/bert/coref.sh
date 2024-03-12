    python experiments/span_train.py \
    -data_path data/ontonotes/coref \
    -task coref -model_type bert \
    -criteria ce \
    -pool_methods max \
    -fine_tune \
    -use_argmax \
    -slurm_comment bert