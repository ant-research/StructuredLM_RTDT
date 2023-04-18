from datasets import load_dataset

dataset = load_dataset(path='openwebtext', name='plain_text', cache_dir='data/openwebtext',
                       download_mode='force_redownload')