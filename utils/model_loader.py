import glob
import torch
import os


def load_model(model, model_path, strict=True):
    state_dict = torch.load(model_path, map_location=lambda a, b: a)
    transfered_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        transfered_state_dict[new_k] = v
    model.load_state_dict(transfered_state_dict, strict=strict)


def load_checkpoint(modules, files, output_dir):
    for module, file in zip(modules, files):
        path = os.path.join(output_dir, file)
        module.load_state_dict(torch.load(path, map_location="cpu"))


def get_max_epoch(output_dir, pattern):
    fn_dir_list = glob.glob(os.path.join(output_dir, pattern))
    if not fn_dir_list:
        return -1

    def get_epoch_num(fn):
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        try:
            epoch = int(fn.replace("model", "").replace(".bin", ""))
            return epoch
        except Exception as e:
            return -1

    epoch_set = set([get_epoch_num(fn) for fn in fn_dir_list])
    if epoch_set:
        return max(epoch_set)
    else:
        return -1

def get_max_epoch_step(output_dir, pattern):
    fn_dir_list = glob.glob(os.path.join(output_dir, pattern))
    if not fn_dir_list:
        return -1, -1

    max_epoch = -1
    max_step = -1
    for fn in fn_dir_list:
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        
        epoch = int(fn.replace("model", "").replace(".bin", "").split('_')[0])
        step = int(fn.replace("model", "").replace(".bin", "").split('_')[1])
        if epoch >= max_epoch:
            if step > max_step or epoch > max_epoch:
                max_epoch = epoch
                max_step = step

    return max_epoch, max_step

def create_r2d2_with_bert_embedding(r2d2_config_path, bert_model, output_path):
    from transformers import AutoConfig
    from model.r2d2_cuda import R2D2Cuda

    config = AutoConfig.from_pretrained(r2d2_config_path)
    model = R2D2Cuda(config)
    bert_state_dict = torch.load(bert_model)
    model.embedding.weight.data = bert_state_dict['bert.embeddings.word_embeddings.weight'].data
    torch.save(model.state_dict(), output_path)
