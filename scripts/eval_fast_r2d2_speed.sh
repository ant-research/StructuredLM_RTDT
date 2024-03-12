python eval/eval_speed.py \
    --model_dir \
    model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --config_path \
    model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
    --vocab_dir \
    model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --corpus_path \
    model_data/en_wiki/wiki103.speed.ids.$1 \
    --max_batch_len \
    2500000 \
    --input_type \
    ids \
    --model \
    fast-r2d2 \
    --turn \
    59 \
    --r2d2_mode \
    $2 \
    --batch_size \
    $3