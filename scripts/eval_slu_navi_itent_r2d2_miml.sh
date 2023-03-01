python -m eval.eval_fast_r2d2_multi_label \
--max_batch_size 8 --max_batch_len 1000000 \
--vocab_dir data/pretrain_dir \
--config_path data/pretrain_dir/config.json \
--model_dir data/save/slu_navigate_intent_r2d2_miml \
--datasets stanfordLU --data_dir data/stanfordLU --turn 19 \
--output_dir data/save/slu_navigate_intent_r2d2_miml_dec --domain navigate \
--task intent --sampler random \
--model_name fastr2d2_miml