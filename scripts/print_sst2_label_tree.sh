python -m eval.r2d2_label_tree_printer \
    --model_path data/save/sst2_noise2/model$1.bin \
    --parser_path data/save/sst2_noise2/parser$1.bin \
    --parser_only \
    --config_path data/pretrain_dir/config.json \
    --corpus_path data/glue/SST-2/train.tsv \
    --output_path data/pred_trees/sst2_label_tree_test_$1.txt \
    --label_num 2 --tsv_column 0 --to_latex_tree