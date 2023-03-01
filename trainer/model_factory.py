import os
from experiments.baseline_models import BertForClassification, BertForDPClassificationMeanPooling, BertForMultiIntent
from model.fast_r2d2_downstream import FastR2D2Classification, FastR2D2CrossSentence
import logging
from model.fast_r2d2_dp_classification import FastR2D2DPClassification, FastR2D2MultiLabelRoot
from experiments.fast_r2d2_miml import FastR2D2MIL, FastR2D2MIML
from experiments.fast_r2d2_parser_encoder import TreeEncMultiLabelWithParser, TreeEncWithParser,ParserTreeEncDP
from utils.model_loader import get_max_epoch, load_model
from transformers import AutoConfig


def create_fast_r2d2_model(args, config_path, label_num, pretrain_dir, output_dir):
    enable_top_down = 'topdown' in args
    enable_exclusive = 'exclusive' in args
    enable_dp = 'dp' in args
    mil = 'mil' in args
    mimll = 'miml' in args
    parser_encoder = 'tree' in args
    
    config = AutoConfig.from_pretrained(config_path)
    if enable_dp and parser_encoder:
        model =  ParserTreeEncDP(config, label_num)
    elif enable_dp:
        model = FastR2D2DPClassification(config, label_num, apply_topdown=enable_top_down, 
                                         exclusive=enable_exclusive)
    elif mil:
        model = FastR2D2MIL(config, label_num)
    elif mimll:
        model = FastR2D2MIML(config, label_num)
    elif parser_encoder:
        model = TreeEncWithParser(config, label_num)
    else:
        model = FastR2D2Classification(config, label_num)

    if pretrain_dir is not None:
        model_path = os.path.join(pretrain_dir, 'model.bin')
        parser_path = os.path.join(pretrain_dir, 'parser.bin')
        model.from_pretrain(model_path, parser_path)

    recover_epoch = -1
    output_dir = None
    if output_dir is not None:
        recover_epoch = get_max_epoch(output_dir, 'model*')
    if recover_epoch >= 0:
        model_recover_checkpoint_r2d2 = os.path.join(output_dir, f"model{recover_epoch}.bin")
        # model_recover_checkpoint_parser = os.path.join(output_dir, f"parser{recover_epoch}.bin")
        # logging.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint_r2d2)
        model.load_model(model_recover_checkpoint_r2d2)
    
    return model


def create_bert_model(args, config_path, label_num, pretrain_dir, output_dir):
    enable_dp = 'dp' in args
    exclusive = 'exclusive' in args
    multi_label = 'multilabel' in args
    if enable_dp:
        model =  BertForDPClassificationMeanPooling(config_path, label_num, exclusive=exclusive)
    else:
        if multi_label:
            model =  BertForMultiIntent(config_path, label_num)
        else:
            model = BertForClassification(config_path, label_num)

    recover_epoch = -1
    if output_dir is not None:
        recover_epoch = get_max_epoch(output_dir, 'model*')
    if recover_epoch >= 0:
        model_recover_checkpoint_r2d2 = os.path.join(output_dir, f"model{recover_epoch}.bin")
        logging.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint_r2d2)
        load_model(model, os.path.join(output_dir, f"model{recover_epoch}.bin"))
    return model

def create_fast_r2d2_cross_sentence(args, config, label_num):
    return FastR2D2CrossSentence(config, label_num)

def create_classification_model(model_name, model_type, config_or_path, label_num, pretrain_dir, output_dir):
    if model_type == 'single':
        args = model_name.split('_')
        if args[0] == 'fastr2d2':
            model = create_fast_r2d2_model(args[1:], config_or_path, label_num, pretrain_dir, output_dir)
        elif args[0] == 'bert':
            model = create_bert_model(args[1:], config_or_path, label_num, pretrain_dir, output_dir)

    elif model_type == 'pair':
        model = create_fast_r2d2_cross_sentence(args[1:], config_or_path, label_num)
    return model


def create_multi_label_classification_model(model_name, config_path, label_num, pretrain_dir):
    args = model_name.lower().split('_')
    config = AutoConfig.from_pretrained(config_path)
    apply_topdown = 'topdown' in args
    exclusive = 'exclusive' in args
    miml = 'miml' in args
    mil = 'mil' in args
    dp = 'dp' in args
    root = 'root' in args
    fp = 'fp' in args
    parser_encoder = 'tree' in args
    if args[0] == 'fastr2d2':
        if miml:
            model = FastR2D2MIML(config, label_num)
        elif mil:
            model = FastR2D2MIL(config, label_num)
        elif parser_encoder and dp:
            model =  ParserTreeEncDP(config, label_num)
        elif fp:
            model =  FastR2D2DPClassification(config, label_num, apply_topdown=apply_topdown, exclusive=exclusive, \
                                              full_permutation=fp)
        elif dp:
            model =  FastR2D2DPClassification(config, label_num, apply_topdown=apply_topdown, exclusive=exclusive)
        elif parser_encoder:
            model = TreeEncMultiLabelWithParser(config, label_num)
        elif root:
            model =  FastR2D2MultiLabelRoot(config, label_num, transformer_parser=False)

    if pretrain_dir is not None:
        model_path = os.path.join(pretrain_dir, 'model.bin')
        parser_path = os.path.join(pretrain_dir, 'parser.bin')
        model.from_pretrain(model_path, parser_path)

    # recover_epoch = None #get_max_epoch_model(args.output_dir)
    # if recover_epoch is not None:
    #     model_recover_checkpoint_r2d2 = os.path.join(args.output_dir, f"model{recover_epoch}.bin")
    #     model_recover_checkpoint_parser = os.path.join(args.output_dir, f"parser{recover_epoch}.bin")
    #     logging.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint_r2d2)
    #     # model.load_model(model_recover_checkpoint_r2d2,model_recover_checkpoint_parser)
    
    return model