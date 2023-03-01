
pid=$!

CUDA_VISIBLE_DEVICES=0 python experiments/miml_evaluator.py --pretrain_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
--dataset atis --data_dir data/ATIS --dataset_mode test --save_path data/save/atis_ner_r2d2_miml/model.bin \
--domain movie_eng --output_path data/save/atis_ner_r2d2_miml --threshold 0.5
kill -9 $pid