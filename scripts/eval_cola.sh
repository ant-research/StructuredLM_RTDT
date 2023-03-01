python -m eval.eval_fast_r2d2 \
    --model_dir \
    data/save/cola_r2d2_dp_readerFixed \
    --config_path \
    data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
    --vocab_dir \
    data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/ \
    --task_type \
    cola \
    --glue_dir \
    data/glue/CoLA \
    --max_batch_len \
    102400 \
    --max_batch_size \
    32 \
    --turn \
    $1 \
    --r2d2_mode \
    forced \
    --model_name fastr2d2_dp \
    --empty_label_idx -1
    # --tree_path data/glue/CoLA/trees.txt