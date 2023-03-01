
pid=$!

python experiments/bert_multi_label_trainer.py \
--max_grad_norm 1 --max_batch_size $((64/$1)) --max_batch_len 1000000 --parser_lr 1e-2 \
--vocab_dir data/bert_12_wiki_103 --config_path data/bert_12_wiki_103 --pretrain_dir data/bert_12_wiki_103 \
--datasets snips --data_dir data/snips --epoch 20 --output_dir data/save/snips_singlelabel_bert_dp_indiceMapFixed \
--log_step 100 --eval_step 30 --task intent --num_samples 256 \
--model_name bert --seed 2023

kill -9 $pid