python -m eval.r2d2_label_tree_printer \
    --model_path data/save/cola_r2d2_dp_exclu_emptyLabelFixed/model$1.bin \
    --parser_only \
    --config_path data/pretrain_dir/config.json \
    --corpus_path data/glue/CoLA/train.tsv \
    --output_path data/pred_trees/cola_label_tree_$1.txt \
    --label_num 2 --tsv_column -1 --to_latex_tree