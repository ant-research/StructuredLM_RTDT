
pid=$!

python trainer/fast_r2d2_multi_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--config_path data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
--pretrain_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--datasets sls --data_dir data/sls --epoch 20 \
--output_dir data/save/sls_restaurant_new_r2d2_dp_topdown_exclu --domain restaurant \
--log_step 100 --eval_step 100 --task intent --sampler random --num_samples 256 \
--model_name fastr2d2_dp --seed 2023

kill -9 $pid