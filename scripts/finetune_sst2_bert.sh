pid=$!

# CUDA_VISIBLE_DEVICES=7 python experiments/bert_glue_trainer.py \
python -m torch.distributed.launch --nproc_per_node $1 --master_port 7095 experiments/bert_glue_trainer.py \
--max_grad_norm 1 --parser_lr 1e-2 --lr 5e-5 --max_batch_size $((64/$1)) --max_batch_len 1000000 --glue_dir data/glue/SST-2 \
--task_type sst-2 --vocab_dir data/bert_12_wiki_103 --config_path data/bert_12_wiki_103 --epoch 20 \
--output_dir data/save/sst2_bert --log_step 100 --eval_step 30 --pretrain_dir data/bert_12_wiki_103 \
--num_samples 256 --sampler random --model_name bert --seed 2023
# --tree_path data/glue/SST-2/trees.txt
sleep 30
kill -9 $pid