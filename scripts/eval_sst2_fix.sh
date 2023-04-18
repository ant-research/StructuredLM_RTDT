python -m eval.eval_fast_r2d2 \
    --model_dir \
    data/save/sst2_r2d2_dp_fix \
    --config_path \
    data/save/sst2_r2d2_dp_fix/config.json \
    --vocab_dir \
    data/save/sst2_r2d2_dp_fix/ \
    --task_type \
    sst-2 \
    --glue_dir \
    data/glue/SST-2 \
    --max_batch_len \
    102400 \
    --max_batch_size \
    32 \
    --turn \
    $1 \
    --model_name fastr2d2_dp_fix