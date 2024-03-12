python -m eval.r2d2_ptb_printer \
    --model_path model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/model.bin \
    --parser_path model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/parser.bin \
    --parser_only --in_word \
    --config_path \
    data/en_config/fast_r2d2.json \
    --corpus_path \
    data/wsj/wsj_valid_raw.txt \
    --output_path \
    pred_trees_par_only_span_inword_4l_dev_$1.txt