nohup python ../watch_dog.py --watch_dir data/save/sst2_postrain --upload_dir workspace/R2D2_save/sst2_postrain/ &

cd trainer

python -m torch.distributed.launch --master_port 12432 --nproc_per_node 8 fast_r2d2_pretrain.py \
    --batch_size 32 --max_batch_len 1024 \
    --lr 5e-5 --parser_lr 1e-2 \
    --vocab_dir ../data/pretrain_dir \
    --config_path ../data/pretrain_dir/config.json \
    --max_grad_norm 1.0 --input_type txt \
    --corpus_path ../data/glue/SST-2/train.raw.txt \
    --output_dir ../data/save/sst2_postrain \
    --num_samples 256 --log_step 100 --epochs 10 \
    --seperator " " --pretrain_dir ../data/pretrain_dir \
    --seed 404