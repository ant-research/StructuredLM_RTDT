# Outline for Summary Dataset Directory
# -DIR_OF_SUMMARY_DATASET
# |-train
# |--0000.parquet
# |--....
# |-valid
# |--0000.parquet
# |-test
# |--0000.parquet

# XSum
python utils/summary_preprocessor.py \
    --mode xsum \
    --summary_dir DIR_OF_SUMMARY_DATASET \
    --output_dir DIR_FOR_PREPROCESSED_DATASET

# CNN-DailyMail
python utils/summary_preprocessor.py \
    --mode cnn \
    --summary_dir DIR_OF_SUMMARY_DATASET \
    --output_dir DIR_FOR_PREPROCESSED_DATASET

# Gigaword
python utils/summary_preprocessor.py \
    --mode gigaword \
    --summary_dir DIR_OF_SUMMARY_DATASET \
    --output_dir DIR_FOR_PREPROCESSED_DATASET
