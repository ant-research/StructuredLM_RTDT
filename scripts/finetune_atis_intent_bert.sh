
pid=$!

# python -m torch.distributed.launch --nproc_per_node $1 --master_port 7088 experiments/bert_multi_label_trainer.py \
python experiments/bert_multi_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/bert_12_wiki_103 --config_path data/bert_12_wiki_103 \
--datasets atis --data_dir data/ATIS --epoch 20 --output_dir data/save/atis_intent_bert_dp_indiceMapFixed --log_step 10 --eval_step 30 \
--pretrain_dir data/bert_12_wiki_103 \
--task intent --num_samples 256 --sampler sequential --model_name bert_multilabel

sleep 30
kill -9 $pid