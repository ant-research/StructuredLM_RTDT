
pid=$!

# python -m torch.distributed.launch --nproc_per_node $1 --master_port 7094 trainer/fast_r2d2_snips_trainer.py \
python trainer/fast_r2d2_single_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--config_path data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
--datasets snips --data_dir data/snips --epoch 20 --output_dir data/save/snips_singlelabel_r2d2_dp_tree_4023 \
--log_step 100 --eval_step 30 --pretrain_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--task intent --num_samples 256 --model_name fastr2d2_dp --seed 4023

sleep 30
kill -9 $pid