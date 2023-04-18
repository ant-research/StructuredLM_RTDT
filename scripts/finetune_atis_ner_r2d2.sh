
pid=$!

# python -m torch.distributed.launch --nproc_per_node $1 --master_port 7095 trainer/fast_r2d2_multi_label_trainer.py \
python trainer/fast_r2d2_multi_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--config_path data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
--datasets atis --data_dir data/ATIS --epoch 20 --output_dir data/save/atis_ner_r2d2_miml_test --log_step 10 --eval_step 30 \
--pretrain_dir data//parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--task NER --num_samples 256 --sampler random --model_name fastr2d2_dp_fix
sleep 30
kill -9 $pid