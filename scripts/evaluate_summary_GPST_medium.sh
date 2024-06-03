# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

start=5
end=9
# Only checkpoints of the last 5 finetune epochs are saved. 
base_path=FINETUNED_CHECKPOINTS_SAVE_DIR

for i in $(seq $start $end)
do
  pretrain_path="${base_path}/model${i}.bin"
  torchrun --standalone --nnodes=1 --nproc_per_node=1 trainer/xsum_trainer_eval.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_512_4_1.json \
    --gpt_config_path data/gpt2-medium/config.json \
    --vocab_dir data/gpt2-medium \
    --summary_dir PATH_TO_GIGAWORD_DATASET \
    --output_dir EVALUATION_SCORE_SAVE_DIR \
    --pretrain_dir "$pretrain_path" \
    --log_step 100 \
    --save_step 1000000000 \
    --epochs 1 \
    --batch_size 64 \
    --eval_batch_size 120 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --coeff_proportion 1.0 \
    --pool_size 3 \
    --document_threshold 400 \
    --summary_threshold 120 \
    --word_sync \
    --eval_perepoch
done