TASK_NAME=CoLA


# python -m torch.distributed.launch --master_port 7097 --nproc_per_node $1 trainer/fast_r2d2_glue_trainer.py \
python experiments/fast_r2d2_glue_trainer_shortcut.py \
    --max_grad_norm 1.0 --lr 5e-5 --parser_lr 1e-2 \
    --vocab_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --config_path data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
    --task_type cola --glue_dir data/glue/$TASK_NAME --max_batch_len 1000000 \
    --max_batch_size $((64/$1)) --output_dir data/save/sst2_r2d2_dp_fix \
    --epochs 20 --pretrain_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --log_step 20 --num_samples -1 --sampler random --apex_mode O1 --empty_label_idx -1 \
    --model_name fastr2d2_dp_topdown --shortcut_type st