    python experiments/span_train.py \
    -data_path data/ontonotes/const/nonterminal \
    -task ctl -model_type bert \
    -criteria bce \
    -pool_methods max \
    -fine_tune \
    -slurm_comment bert