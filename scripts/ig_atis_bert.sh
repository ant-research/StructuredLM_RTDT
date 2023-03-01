
pid=$!

CUDA_VISIBLE_DEVICES=0 python experiments/integrated_gradient.py \
--dataset atis --data_dir data/ATIS --dataset_mode test --save_path data/save/atis_ner_bert_wiki103/model.bin \
--domain movie_eng --output_path data/save/atis_ner_bert_wiki103 --threshold 0.6 --span_bucket 1,2,3,5
kill -9 $pid