# For WikiText-103, raw_corpus_path should be path to wiki.train.txt, e.g.: corpus/wiki.train.txt; 
#   output_path should be a directory named ended with .lazy
# For OpenWebText, raw_corpus_path should be path to openwebtext directory e.g. corpus/openwebtext; 
#   output_path should be a directory named ended with .lazy

# Outline for OpenWebText Directory
# -OpenWebText Directory
# |-urlsf_subset00-1_data
# |-....
# |-urlsf_subset20-99_data


# WikiText-103
python utils/dataset_builder.py \
    --mode wikitext103 \
    --tokenizer_config_path PATH_TO_TOKENIZER_CONFIG \
    --raw_corpus_path PATH_TO_RAW_CORPUS \
    --output_path PATH_TO_OUTPUT_DIR

# OpenWebText
python utils/dataset_builder.py \
    --mode openwebtext \
    --tokenizer_config_path PATH_TO_TOKENIZER_CONFIG \
    --raw_corpus_path PATH_TO_RAW_CORPUS \
    --output_path PATH_TO_OUTPUT_DIR

# python utils/dataset_builder.py \
#     --mode wikitext103 \
#     --tokenizer_config_path data/gpt2-small \
#     --raw_corpus_path corpus/wiki103/wiki.train.txt \
#     --output_path corpus/wiki103.lazy

# python utils/dataset_builder.py \
#     --mode openwebtext \
#     --tokenizer_config_path data/gpt2-small \
#     --raw_corpus_path corpus/openwebtext \
#     --output_path corpus/openwebtext.lazy
