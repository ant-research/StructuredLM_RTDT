
pid=$!

CUDA_VISIBLE_DEVICES=7 python experiments/integrated_gradient.py \
--dataset sls --data_dir data/sls --dataset_mode test --save_path data/save/sls_restaurant_new_bert_wiki103/model.bin \
--domain restaurant --output_path data/save/sls_restaurant_new_bert_wiki103 --threshold 0.6
kill -9 $pid