python -m eval.eval_fast_r2d2 \
    --model_dir \
    model_data/fintune_sst_2_atomspan_4l_60_B64 \
    --config_path \
    model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
    --vocab_dir \
    model_data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --task_type \
    sst-2 \
    --glue_dir \
    model_data/glue/SST-2 \
    --max_batch_len \
    1024 \
    --max_batch_size \
    32 \
    --turn \
    $1 \
    --r2d2_mode \
    $2