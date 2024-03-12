#!/bin/bash

cd eval

start=0
end=9

for ((i=start;i<=end;i++)); do
    python r2d2plus_ptb_printer.py --model_path ../data/MODEL_DIR/model${i}.bin \
        --parser_path ../data/MODEL_DIR/parser${i}_39.bin \
        --parser_only --config_path ../data/en_config/MODEL_CONFIG \
        --corpus_path ../data/wsj/wsj_valid_raw.txt --in_word --output_path ../data/OUTPUT_DIR/valid_output_epoch${i}.txt
done

for ((i=start;i<=end;i++)); do
    python r2d2plus_ptb_printer.py --model_path ../data/MODEL_DIR/model${i}.bin \
        --parser_path ../data/posttrain_wsj_noshare_mlpshare_win4/parser${i}.bin \
        --parser_only --config_path ../data/en_config/MODEL_CONFIG \
        --corpus_path ../data/wsj/wsj_test_raw.txt --in_word --output_path ../data/OUTPUT_DIR/test_output_epoch${i}.txt
done