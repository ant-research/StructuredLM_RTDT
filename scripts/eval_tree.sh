python -m eval.r2d2_ptb_printer \
    --model_path data/save/wiki_wsj_lstm_rand_span/model.bin \
    --parser_path data/save/wiki_wsj_lstm_rand_span/parser$1.bin \
    --parser_only \
    --config_path data/save/wiki_wsj_lstm_rand_span/config.json \
    --corpus_path data/wsj/wsj_test_raw.txt \
    --in_word \
    --output_path data/pred_trees/wsj_rand_span_$1.txt

python -m eval.compare_tree --tree1 data/wsj/ptb-test.txt --tree2 data/pred_trees/wsj_rand_span_$1.txt