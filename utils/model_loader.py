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
        epoch = int(fn.replace("model", "").replace(".bin", ""))
        return epoch

    epoch_set = set([get_epoch_num(fn) for fn in fn_dir_list])
    if epoch_set:
        return max(epoch_set)
    else:
        return -1