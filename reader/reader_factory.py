from reader.glue_reader import GlueReaderForDP, GlueReaderWithNoise, R2D2GlueReader


def create_glue_dataset(tokenizer, enable_dp, task_type, glue_dir, dataset_type, 
                        max_batch_len, max_batch_size, sampler, 
                        noise_corpus=None, empty_label_idx=-1,
                        tree_path=None):
    if not enable_dp:
        dataset = R2D2GlueReader(
            task_type,
            glue_dir,
            dataset_type,
            tokenizer,
            max_batch_len=max_batch_len,
            max_batch_size=max_batch_size,
            random=sampler == "random",
            seperator=" ",
            tree_path=tree_path
        )
    else:
        if noise_corpus is None:
            dataset = GlueReaderForDP(
                task_type,
                glue_dir,
                dataset_type,
                tokenizer,
                max_batch_len=max_batch_len,
                max_batch_size=max_batch_size,
                random=sampler == "random",
                empty_label_idx=empty_label_idx,
                seperator=" ",
                tree_path=tree_path
            )
        else:
            dataset = GlueReaderWithNoise(
                task_type,
                glue_dir,
                noise_corpus,
                dataset_type,
                tokenizer,
                max_batch_len=max_batch_len,
                max_batch_size=max_batch_size,
                random=sampler == "random",
                empty_label_idx=empty_label_idx
            )
    return dataset