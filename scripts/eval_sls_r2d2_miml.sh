python -m eval.eval_fast_r2d2_multi_label \
--max_batch_size 8 --max_batch_len 1000000 \
--vocab_dir data/pretrain_dir \
--config_path data/pretrain_dir/config.json \
--model_dir data/save/fast_r2d2_sls_miml \
--datasets sls --data_dir data/sls --turn 19 \
--output_dir data/save/fast_r2d2_sls_miml --domain movie_eng \
--task NER --sampler random \
--model_name fastr2d2_miml