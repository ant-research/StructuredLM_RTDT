from shutil import copy
import os
from collections import OrderedDict
import hashlib


def compute_name(args, imp_opts):
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    hp_dict = vars(args)
    for key in imp_opts:
        opt_dict[key] = hp_dict[key]

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = args.task + '_' + args.slurm_comment + '_' + str(hash_idx)
    if args.fine_tune:
        model_name = "ft_" + model_name
    return model_name


def get_model_path(exp_path, args):
    # because mlp-diora is default for many of the experiments without dtype as their args, so this function is needed to standardize the name
    imp_opts = [
        'task', 'data_path',
        'batch_size', 'real_batch_size', 'eval_batch_size', 'epochs',
        'optimizer', 'learning_rate',
        'eval_step',
        'seed',
        'train_frac', 'train_length_filter',  # train filter
        'span_dim', 'use_proj',  # dim
        'model_type', 'model_size', 'cased',  # Encoder
        'pool_methods',  # SpanRepr
        'fine_tune', 
        'criteria'
    ]
    name = compute_name(args, imp_opts)
    model_path = os.path.join(exp_path, name)
    if not os.path.exists(os.path.join(model_path, 'ckpt')):
        if args.eval:
            raise NotImplementedError("No trained model exist!")
        else:
            os.makedirs(model_path, exist_ok=True)
    return model_path
