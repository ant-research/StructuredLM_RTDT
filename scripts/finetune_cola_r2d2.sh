TASK_NAME=CoLA

# python -m torch.distributed.launch --master_port 7093 --nproc_per_node $1 trainer/fast_r2d2_glue_trainer.py \
python trainer/fast_r2d2_glue_trainer.py \
    --max_grad_norm 1.0 --lr 5e-5 --parser_lr 1e-2 \
    --vocab_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --config_path data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60/config.json \
    --task_type cola --glue_dir data/glue/$TASK_NAME --max_batch_len 1000000 --enable_epoch_eval \
    --max_batch_size $((64/$1)) --output_dir data/save/cola_r2d2_tree_readerFixed_shuffle \
    --epochs 20 --pretrain_dir data/parser_atomspan_r2d2_4l_notie_wiki103wash_a100_60 \
    --log_step 20 --num_samples 1 --sampler random --apex_mode O1 --empty_label_idx -1 \
    --model_name fastr2d2_dp --seed 2023
    #  --tree_path data/glue/CoLA/trees.txt