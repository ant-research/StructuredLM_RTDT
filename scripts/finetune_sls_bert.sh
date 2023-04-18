
pid=$!

# python -m torch.distributed.launch --nproc_per_node $1 --master_port 7092 experiments/bert_multi_label_trainer.py \
python experiments/bert_multi_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/bert_12_wiki_103 \
--config_path data/bert_12_wiki_103 \
--pretrain_dir data/bert_12_wiki_103 \
--datasets sls --data_dir data/sls --epoch 20 \
--output_dir data/save/sls_restaurant_new_bert_wiki103 --domain restaurant \
--log_step 100 --eval_step 100 --task NER --sampler random --num_samples 256 \
--model_name bert_multilabel --seed 2023
kill -9 $pid