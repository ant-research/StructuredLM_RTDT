# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

import os
import logging
from utils.model_loader import get_max_epoch, load_model
from transformers import AutoConfig

    
def create_fast_r2d2_plus_model(model_name, config_path, label_num, pretrain_dir, output_dir):
    # TODO: use output_dir for resuming former finetuning
    from experiments.fast_r2d2_io_downstream import FastR2D2IOClassification

    config = AutoConfig.from_pretrained(config_path)
    if model_name == 'io': # FastR2D2 + inside-outside
        return FastR2D2IOClassification(config, label_num, pretrain_dir=pretrain_dir)
    elif model_name == 'iter': # FastR2D2 + iter
        from experiments.fast_r2d2_iter_downstream import FastR2D2IterClassification
        return FastR2D2IterClassification(config, label_num, pretrain_dir=pretrain_dir)
    elif model_name == 'noattn': # FastR2D2 + iter
        from experiments.fast_r2d2_iter_downstream_abl import FastR2D2IterClassification
        return FastR2D2IterClassification(config, label_num, pretrain_dir=pretrain_dir)
    elif model_name == 'term':
        from experiments.fast_r2d2_iter_downstream_terminal_only import FastR2D2IterClassification
        return FastR2D2IterClassification(config, label_num, pretrain_dir=pretrain_dir)


def create_classification_model(model_name, model_type, config_or_path, label_num, pretrain_dir, output_dir=None):
    if model_type == 'single':
        args = model_name.split('_')
        elif args[0] == 'fastr2d2+':
            model = create_fast_r2d2_plus_model(args[1], config_or_path, label_num, pretrain_dir, output_dir)
        elif args[0] == 'transformer':
            model = create_transformer_model(config_or_path, label_num, pretrain_dir)

    elif model_type == 'pair':
        args = model_name.split('_')
        elif args[0] == 'fastr2d2+':
            model = create_fast_r2d2_plus_model(args[1], config_or_path, label_num, pretrain_dir, output_dir)
        elif args[0] == 'transformer':
            model = create_transformer_model(config_or_path, label_num, pretrain_dir)
        else:
            raise Exception(f'unsupported model_type: {model_type}')
    return model

def create_transformer_model(config_path, label_num, pretrain_dir):
    from experiments.transformer_downstream import TransformerDownstream
    config = AutoConfig.from_pretrained(config_path)
    model = TransformerDownstream(config, label_num)
    if pretrain_dir is not None:
        model.from_pretrain(pretrain_dir)
    return model