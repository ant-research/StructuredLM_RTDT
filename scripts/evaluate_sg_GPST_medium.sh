python eval/sg_evaluator.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_512_4_1.json \
    --gpt_config_path data/gpt2-medium/config.json \
    --vocab_dir data/gpt2-medium \
    --pretrain_dir PATH_TO_PRETRAINED_MODEL \
    --task_dir DIR_OF_PREPROCESSED_SG_DATA \
    --alltest