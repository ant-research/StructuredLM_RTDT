# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Qingyang Zhu

import argparse
import sys
from functools import reduce
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.pretrained_transformers import Encoder
from encoders.pretrained_transformers.simple_encoder import SimpleEncoder
from encoders.pure_transformer_wrapper import PureTransformerWrapper
from experiments.fast_r2d2_iter_downstream import FastR2D2IterSpanClassification, FastR2D2SpanClassification
from span_model import SpanModel
from span_data import SpanDataset
from span_utils import instance_f1_info, f1_score, print_example
from utils.tree_utils import get_tree_from_merge_trajectory

from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from util.iterator import FixLengthLoader
from util.logger import configure_logger, get_logger
from util.name import get_model_path


class LearningRateController(object):
    # Learning rate controller copied form constituent/train.py
    def __init__(self, weight_decay_range=5, terminate_range=20):
        self.data = list()
        self.not_improved = 0
        self.weight_decay_range = weight_decay_range
        self.terminate_range = terminate_range
        self.best_performance = -1e10

    def add_value(self, val):
        # add value
        if len(self.data) == 0 or val > self.best_performance:
            self.not_improved = 0
            self.best_performance = val
        else:
            self.not_improved += 1
        self.data.append(val)
        return self.not_improved


def forward_batch(task, model, batch, mode='loss', use_argmax=None, num_label=None):
    """
    NOTE:
        For current ver. the training loss is integrated with model loss.
    """
    labels_3d = batch['labels']
    preds = model(batch)
    if isinstance(preds, dict): # r2d2 mode
        out_dict = preds
        preds = out_dict['preds']
        model_loss = out_dict['model_loss']
        trees_dict = out_dict['trees_dict']
    else: # transformer mode
        model_loss = None
        trees_dict = None

    num_pred = preds.shape[0]
    
    if hasattr(model, "label_num"):
        num_label = model.label_num
    else:
        num_label = len(model.label_itos)

    one_hot_labels = torch.zeros(num_pred, num_label).long()

    def flatten_list(input_list):
        return reduce(lambda xs, x: xs + x, input_list, [])

    labels_2d = flatten_list(labels_3d)
    labels_1d = flatten_list(labels_2d)

    span_idx = reduce(lambda xs, i: xs + [i] * len(labels_2d[i]), range(num_pred), [])
    one_hot_labels[span_idx, labels_1d] = 1

    if torch.cuda.is_available():
        one_hot_labels = one_hot_labels.cuda()

    '''
    there are two ways of generating answers
    one is to pick the label value > 0.5 
    one is to pick the most possible label
    in some tasks like ctl, there might be multiple labels for one span
    '''
    if model.criteria != 'ce': # BCELoss
        if use_argmax:
            p = torch.argmax(preds, dim=1).cuda()
            pred_labels = torch.zeros_like(preds)
            pred_labels.scatter_(1, p.unsqueeze(dim=1), 1)
            pred_labels = pred_labels.long()
        else:
            preds_probs = torch.sigmoid(preds)
            pred_labels = (preds_probs > 0.5).long()
        
        if mode == 'pred_loss':
            loss = model.training_criterion(preds, one_hot_labels.float())
            return pred_labels, one_hot_labels, loss, model_loss
        elif mode == 'pred':  # for validation
            return pred_labels, one_hot_labels
        elif mode == 'loss':  # for training
            loss = model.training_criterion(preds, one_hot_labels.float())
            return loss, model_loss, trees_dict
    else:
        m = nn.Softmax(dim=1)
        new_preds = m(preds)
        p = torch.argmax(new_preds, dim=1).cuda()
        pred_labels = torch.zeros_like(new_preds)
        pred_labels.scatter_(1, p.unsqueeze(dim=1), 1)
        pred_labels = pred_labels.long()

        l = torch.tensor(labels_1d).cuda()
            
        if mode == 'pred_loss':
            loss = F.cross_entropy(preds, l)
            return pred_labels, one_hot_labels, loss, model_loss
        elif mode == 'pred':  # for validation
            return pred_labels, one_hot_labels
        elif mode == 'loss':  # for training
            loss = F.cross_entropy(preds, l)
            return loss, model_loss, trees_dict

def validate(task, loader, model, getloss=False, output_example=False, use_argmax=False):
    # save the random state for recovery
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.random.get_rng_state()
    numerator = denom_p = denom_r = 0

    if getloss == True:
        cumulated_loss = cumulated_num = 0
        cumulated_model_loss = 0
        for i, batch_dict in enumerate(loader):
        
            preds, ans, loss, model_loss = forward_batch(task, model, batch_dict, mode='pred_loss', use_argmax=use_argmax)

            if task == "ner":
                num_pred = preds.shape[0]
                num_label = len(model.label_itos)
                mask = torch.ones(num_pred, num_label).long()

                if torch.cuda.is_available():
                    mask = mask.cuda()

                mask[:, model.label_stoi["none"]] = 0
                ans = ans * mask
                preds = preds * mask

            num_instances = len(batch_dict['labels'])
            cumulated_loss += loss.item() * num_instances
            if model_loss:
                cumulated_model_loss += (model_loss[0].item() + model_loss[1].item()) * num_instances
            cumulated_num += num_instances
            num, dp, dr = instance_f1_info(ans, preds)
            numerator += num
            denom_p += dp
            denom_r += dr
        val_loss = cumulated_loss / cumulated_num
        if cumulated_model_loss:
            val_model_loss = cumulated_model_loss /cumulated_num
        else:
            val_model_loss = None
        # recover the random state for reproduction
        torch.random.set_rng_state(rng_state)
        torch.cuda.random.set_rng_state(cuda_rng_state)
        return f1_score(numerator, denom_p, denom_r), val_loss, val_model_loss
    else:
        for batch_dict in loader:
        
            preds, ans = forward_batch(task, model, batch_dict, mode='pred', use_argmax=use_argmax)
            
            if task == "ner":
                num_pred = preds.shape[0]
                num_label = len(model.label_itos)
                mask = torch.ones(num_pred, num_label).long()

                if torch.cuda.is_available():
                    mask = mask.cuda()

                mask[:, model.label_stoi["none"]] = 0
                ans = ans * mask
                preds = preds * mask

            num, dp, dr = instance_f1_info(ans, preds)
            numerator += num
            denom_p += dp
            denom_r += dr

        # recover the random state for reproduction
        torch.random.set_rng_state(rng_state)
        torch.cuda.random.set_rng_state(cuda_rng_state)
        return f1_score(numerator, denom_p, denom_r)


def log_arguments(args):
    # log the parameters
    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")


def set_seed(seed):
    # initialize random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_parser():
    # arguments from snippets
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('-data_path', type=str, default='data/ontonotes/ner')
    parser.add_argument('-exp_path', type=str, default='./out')
    parser.add_argument('-config_path', type=str, default='data/r2d2+_noshare_30')
    parser.add_argument('-pretrain_dir', type=str, default=None)
    parser.add_argument('-vocab_dir', type=str, default='data/en_config')
    # shortcuts
    # experiment type
    parser.add_argument('-task', type=str, default='nel', choices=('nel', 'ctl', 'coref', 'src', 'ctd', 'med', 'ner'))
    # training setting
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-real_batch_size', type=int, default=128)
    parser.add_argument('-eval_batch_size', type=int, default=32)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-optimizer', type=str, default='Adam')
    parser.add_argument('-learning_rate', type=float, default=5e-4)
    parser.add_argument("-attn_lr", type=float, default=2e-4)
    parser.add_argument("-encoder_lr", type=float, default=5e-5)
    parser.add_argument("-parser_lr", type=float, default=1e-4)
    parser.add_argument('-log_step', type=int, default=50)
    parser.add_argument('-eval_step', type=int, default=500)
    parser.add_argument('-criteria', type=str, default='bce', choices=('bce', 'ce'))
    parser.add_argument('-seed', type=int, default=4)
    parser.add_argument('-train_length_filter', type=int, default=1000)
    parser.add_argument('-eval_length_filter', type=int, default=1000)
    parser.add_argument('-weight_decay_range', type=int, default=5)
    parser.add_argument('-mlm_rate', type=float, default=0.0) # 0.15 if mlm
    parser.add_argument('-decline_rate', type=float, default=0.0) # 0.015 if mlm
    # customized arguments
    parser.add_argument('-span_dim', type=int, default=256)
    parser.add_argument('-use_proj', action='store_true', default=False)

    # encoder arguments
    parser.add_argument('-model_type', type=str, default='r2d2',
                        choices=('bert', 'transformer', 'r2d2', 'fastr2d2'))
    parser.add_argument('-share', default=False, action='store_true',
                        help='whether share up & down params in r2d2')
    parser.add_argument('-model_size', type=str, default='base')
    parser.add_argument('-uncased', action='store_false', dest='cased')

    # pool_method
    parser.add_argument('-pool_methods', type=str, nargs="*", default='max',
                        choices=('mean', 'max', 'diff_sum', 'endpoint', 'attn'))

    # span attention
    parser.add_argument('-attn_schema', nargs='+', type=str, default=['none'])
    parser.add_argument("-nhead", type=int, default=2)
    parser.add_argument("-nlayer", type=int, default=2)

    parser.add_argument('-fine_tune', action='store_true', default=False)


    # args for test
    parser.add_argument('-train_frac', default=1.0, type=float)
    parser.add_argument('-eval', action='store_true', default=False)
    parser.add_argument('-disable_loading', default=False, action='store_true',
                        help='Not to load from existing checkpoints')
    parser.add_argument('-output_example', default=False, action='store_true',
                        help='Output the incorrect results')
    parser.add_argument('-use_argmax', default=False, action='store_true',
                        help='Use argmax instead of requiring the softmax score to be > 0.5')
    parser.add_argument('-output_rp', default=False, action='store_true',
                        help='Output recall and precision')
    parser.add_argument('-time_limit', type=float, default=288000, help='Default time limit: 80 hours')
    parser.add_argument('-slurm_comment' , type=str, default="none")

    return parser


def process_args(args):
    # For convenience of setting path args.
    for k, v in args.__dict__.items():
        if type(v) == str and v.startswith('~'):
            args.__dict__[k] = os.path.expanduser(v)
    return args


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = process_args(args)

    set_seed(args.seed)
    if args.task in ('ctl', 'nel', 'ctd', 'med', 'ner'):
        num_spans = 1
    elif args.task in ('coref', 'src'):
        num_spans = 2
    else:
        raise NotImplementedError()
    # save arguments
    model_path = get_model_path(args.exp_path, args)
    log_path = os.path.join(model_path, "log")
    if not args.eval:
        configure_logger(log_path)
        log_arguments(args)
    logger = get_logger()

    args.start_time = time.time()
    logger.info(f"Model path: {model_path}")


    #####################
    # create data sets, tokenizers, and data loaders
    #####################
    # Set whether fine tune token encoder.
    encoder_dict = {}
    args.pool_methods = args.pool_methods
    if args.model_type == 'bert':
        encoder_dict[args.model_type] = Encoder(args.model_type, args.model_size, args.cased,
                                                fine_tune=args.fine_tune)
    elif args.model_type == 'r2d2':
        config = AutoConfig.from_pretrained(args.config_path)
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
        encoder_dict[args.model_type] = SimpleEncoder(tokenizer)
       
    elif args.model_type == 'fastr2d2':
        config = AutoConfig.from_pretrained(args.config_path)
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
        encoder_dict[args.model_type] = SimpleEncoder(tokenizer)
        
    elif args.model_type == 'transformer':
        config = AutoConfig.from_pretrained(args.config_path)
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
        encoder = PureTransformerWrapper(config, tokenizer, args.pretrain_dir)
        encoder_dict[args.model_type] = encoder
        
    else:
        raise NotImplementedError()
    
    data_loader_path = os.path.join(model_path, 'dataloader.pt')
    # TODO:
    use_word_level_span_idx = any([("iornn" in pm) or ("diora" in pm) for pm in args.pool_methods]) # not used
    mask_id = encoder_dict[args.model_type].tokenizer.convert_tokens_to_ids('[MASK]')

    if args.eval:
    # if in eval mode, we only need to load the test set
        logger.info('Creating datasets in eval mode.')
        try:
            data_info = torch.load(data_loader_path)
            SpanDataset.label_dict = data_info['label_dict']
        except:  # dataloader do not exist or dataloader is outdated
        # to create label_dict by initializing a SpanDataset
            s = SpanDataset(
                os.path.join(args.data_path, 'train.json'),
                encoder_dict=encoder_dict,
                train_frac=args.train_frac,
                length_filter=args.train_length_filter,
                word_level_span_idx=use_word_level_span_idx
            )
        data_set = SpanDataset(
            os.path.join(args.data_path, 'test.json'),
            encoder_dict=encoder_dict,
            length_filter=args.eval_length_filter,
            word_level_span_idx=use_word_level_span_idx
        )
        
        data_loader = FixLengthLoader(data_set, args.eval_batch_size, shuffle=False,
                                      mask_id=mask_id)

    elif os.path.exists(data_loader_path) and not args.disable_loading:
    # if dataloader exists and we are not in eval mode, we load the dataloader
        logger.info('Loading datasets.')
        data_info = torch.load(data_loader_path)
        data_loader = data_info['data_loader']

        for split in ['train', 'development', 'test']:
            is_train = (split == 'train')
            bs = args.batch_size if is_train else args.eval_batch_size
            mlm_rate = args.mlm_rate if split == 'train' else 0.0 # eval no mask
            data_loader[split] = FixLengthLoader(data_loader[split].dataset, bs, shuffle=is_train,
                                                 mask_id=mask_id,
                                                 mlm_rate=mlm_rate, decline_rate=args.decline_rate)
        SpanDataset.label_dict = data_info['label_dict']
    else:
    # if dataloader does not exist, we create the dataloader
        logger.info("Creating datasets from: %s" % args.data_path)
        data_set = dict()
        data_loader = dict()
        for split in ['train', 'development', 'test']:
            is_train = (split == 'train')
            frac = args.train_frac if is_train else 1.0
            len_filter = args.train_length_filter if is_train else args.eval_length_filter
            bs = args.batch_size if is_train else args.eval_batch_size
            data_set[split] = SpanDataset(
                os.path.join(args.data_path, f'{split}.json'),
                encoder_dict=encoder_dict,
                train_frac=frac,
                length_filter=len_filter,
                word_level_span_idx=use_word_level_span_idx
            )
            mlm_rate = args.mlm_rate if split == 'train' else 0.0 # eval no mask
            data_loader[split] = FixLengthLoader(data_set[split], bs, shuffle=is_train,
                                                 mask_id=mask_id,
                                                 mlm_rate=mlm_rate, decline_rate=args.decline_rate)

        torch.save(
            {
                'data_loader': data_loader,
                'label_dict': SpanDataset.label_dict
            },
            data_loader_path
        )

    logger.info("Dataset info:")
    logger.info('-' * 80)
    if not args.eval:
        for split in ('train', 'development', 'test'):
            logger.info(split)
            dataset = data_loader[split].dataset
            for k in dataset.info:
                logger.info(f'{k}:{dataset.info[k]}')
            logger.info('-' * 80)
    else:
        logger.info('test')
        dataset = data_loader.dataset
        for k in dataset.info:
            logger.info(f'{k}:{dataset.info[k]}')
        logger.info('-' * 80)

    # initialize model
    logger.info('Initializing models.')
    if args.model_type == 'r2d2':
        model = FastR2D2IterSpanClassification(config, len(data_loader['train'].dataset.label_dict), 
                                               transformer_parser=False,
                                               pretrain_dir=args.pretrain_dir,
                                               finetune_parser=True,
                                               num_repr=num_spans, 
                                               tokenizer=encoder_dict[args.model_type].tokenizer,
                                               criteria=args.criteria,
                                               share=args.share
                                               )
    elif args.model_type ==  'fastr2d2':
        model = FastR2D2SpanClassification(config, len(data_loader['train'].dataset.label_dict),
                                           pretrain_dir=args.pretrain_dir, num_repr=num_spans,
                                           tokenizer=encoder_dict[args.model_type].tokenizer,
                                           criteria=args.criteria)
                                           
    else:
        model = SpanModel(
            encoder_dict, span_dim=args.span_dim, pool_methods=args.pool_methods, use_proj=args.use_proj, 
            attn_schema=args.attn_schema, nhead=args.nhead, nlayer=args.nlayer, 
            label_itos={value: key for key, value in SpanDataset.label_dict.items()},
            label_stoi={key: value for key, value in SpanDataset.label_dict.items()},
            criteria = args.criteria,
            num_spans=num_spans
        )
    label_num = len(SpanDataset.label_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    # initialize optimizer
    if not args.eval:
        logger.info('Initializing optimizer.')

        logger.info('Fine tune information: ')
        if args.fine_tune:
            logger.info('Fine tuning parameters in Encoder')

        logger.info('Trainable parameters: ')
        if args.model_type in ['transformer', 'bert'] : # transformer based
            logger.info('transformer-based .')
            params = list()
            encoder_params = list()
            attn_params = list()
            names = list()
            for name, param in list(model.named_parameters()):
                if param.requires_grad:
                    if 'trans' not in name:
                        if 'encoder' in name:
                            encoder_params.append(param)
                        else:
                            params.append(param)
                    else:
                        attn_params.append(param)
                    names.append(name)
            optimizer = getattr(torch.optim, args.optimizer)([{'params': params, 'lr': args.learning_rate},
                                                            {'params': encoder_params, 'lr': args.encoder_lr},
                                                            {'params': attn_params, 'lr': args.attn_lr}])
        else: # r2d2 based
            logger.info('r2d2-based .')
            parser_params = []
            r2d2_params = []
            other_params = []
            for name, params in model.named_parameters():
                if params.requires_grad:
                    if name.startswith('parser'):
                        parser_params.append(params)
                    elif name.startswith('r2d2'):
                        r2d2_params.append(params)
                    else:
                        other_params.append(params)
            
            optimizer = getattr(torch.optim, args.optimizer)([{'params': other_params, 'lr': args.learning_rate},
                                                        {'params': r2d2_params, 'lr': args.encoder_lr},
                                                        {'params': parser_params, 'lr': args.parser_lr}])
    # initialize best model info, and lr controller
    best_f1 = 0
    best_model = None
    lr_controller = LearningRateController(weight_decay_range=args.weight_decay_range)
    scaler = torch.cuda.amp.GradScaler()

    # load checkpoint, if exists
    args.start_epoch = 0
    args.epoch_step = -1
    ckpt_path = os.path.join(model_path, 'ckpt')

    if args.eval:
        checkpoint = torch.load(ckpt_path)
        best_model = checkpoint['best_model']
        assert best_model is not None
        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            test_f1 = validate(args.task, data_loader, model, output_example=args.output_example, use_argmax=args.use_argmax)
            logger.info(f'Test F1 {test_f1 * 100:6.2f}%')
        return 0

    if os.path.exists(ckpt_path) and not args.disable_loading:
        logger.info(f'Loading checkpoint from {ckpt_path}.')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        best_model = checkpoint['best_model']
        best_f1 = checkpoint['best_f1']
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
        lr_controller = checkpoint['lr_controller']
        scaler.load_state_dict(checkpoint['scaler'])
        torch.cuda.random.set_rng_state(checkpoint['cuda_rng_state'])
        args.start_epoch = checkpoint['epoch']
        args.epoch_step = checkpoint['step']

    logger.info('start training ...')
    model.eval()
    logger.info('-' * 80)
    with torch.no_grad():
        curr_f1 = validate(args.task, data_loader['development'], model, use_argmax=args.use_argmax)
    logger.info(f'Validation F1 {curr_f1 * 100:6.2f}%')

    # training
    MAX_GRAD_NORM = 1.0
    terminate = False
    for epoch in range(args.epochs):
        if terminate:
            break
        model.train()
        cumulated_loss = cumulated_num = 0
        cumulated_mlm_loss = cumulated_kl_loss = 0
        data_loader['train'].set_epoch(epoch) # update mask rate wrt epoch # (MLM training warmup)
        for step, batch in enumerate(data_loader['train']):
            if terminate:
                break
            # ignore batches to recover the same data loader state of checkpoint
            if (epoch < args.start_epoch) or (epoch == args.start_epoch and step <= args.epoch_step):
                continue
                    
            with torch.cuda.amp.autocast():
                loss, model_loss, trees_dict = forward_batch(args.task, model, batch, mode='loss', use_argmax=args.use_argmax)
                
            actual_step = len(data_loader['train']) * epoch + step + 1
            # optimize model with gradient accumulation
            if (actual_step - 1) % (args.real_batch_size // args.batch_size) == 0:
                optimizer.zero_grad()
                
            total_loss = loss + sum(model_loss) if model_loss else loss
            try:
                scaler.scale(total_loss).backward()
                if actual_step % (args.real_batch_size // args.batch_size) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
            except RuntimeError as e:
                logger.error(e)

            # update metadata
            num_instances = len(batch['labels'])
            cumulated_loss += loss.item() * num_instances
            if model_loss:
                cumulated_mlm_loss += model_loss[0].item() * num_instances
                cumulated_kl_loss += model_loss[1].item() * num_instances
            cumulated_num += num_instances
            # log
            if (actual_step % (args.real_batch_size // args.batch_size) == 0) and (
                    actual_step // (args.real_batch_size // args.batch_size)) % args.log_step == 0:
                if args.model_type == "r2d2":
                    # print learned parse trees if r2d2
                    with torch.no_grad():
                        model.eval()
                        input_ids = batch['subwords']['r2d2'][0]
                        attention_mask = (input_ids != model.tokenizer.pad_token_id).int()
                    
                        seq_len = attention_mask.sum()
                        tokens = model.tokenizer.convert_ids_to_tokens(input_ids.cpu().data.numpy())
                        if trees_dict:
                            split_points = [_ for _ in reversed(
                                        trees_dict['split_points'][0, 0, :].cpu().data.numpy()[:seq_len])]
                            merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)[1]
                            logger.info(f"parsed tree : {merged_tree}")
                        else:
                            logger.info(f"input token: {' '.join(tokens)}")
                            
                        model.train()
                if model_loss:
                    logger.info(
                        f'Train '
                        f'Epoch #{epoch} | Step {actual_step // (args.real_batch_size // args.batch_size)} | '
                        f'pred loss {cumulated_loss / cumulated_num:8.4f} | '
                        f'mlm loss {cumulated_mlm_loss / cumulated_num:8.4f} | '
                        f'kl loss {cumulated_kl_loss / cumulated_num:8.4f}' 
                    )
                else:
                    logger.info(
                        f'Train '
                        f'Epoch #{epoch} | Step {actual_step // (args.real_batch_size // args.batch_size)} | '
                        f'pred loss {cumulated_loss / cumulated_num:8.4f}'
                    )
            # validate
            if (actual_step % (args.real_batch_size // args.batch_size) == 0) and (
                    actual_step // (args.real_batch_size // args.batch_size)) % args.eval_step == 0:
            # if True:
                model.eval()
                logger.info('-' * 80)
                with torch.no_grad():
                    curr_f1, val_loss_step, val_model_loss_step = validate(args.task, data_loader['development'], model, getloss=True, use_argmax=args.use_argmax)
                logger.info(f'Validation F1 {curr_f1 * 100:6.2f}%')
                # update when there is a new best model
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    best_model = model.state_dict()
                    logger.info('New best model!')
                logger.info('-' * 80)
                model.train()
                # update validation result
                not_improved_epoch = lr_controller.add_value(curr_f1)
                if not_improved_epoch == 0:
                    pass
                elif not_improved_epoch >= lr_controller.terminate_range:
                    logger.info(
                        'Terminating due to lack of validation improvement.')
                    terminate = True
                elif not_improved_epoch % lr_controller.weight_decay_range == 0:
                    logger.info(
                        f'Re-initialize learning rate to '
                        f'{optimizer.param_groups[0]["lr"] / 2.0:.8f}, {optimizer.param_groups[1]["lr"] / 2.0:.8f}, {optimizer.param_groups[2]["lr"] / 2.0:.8f}'
                    )
                    if args.model_type in ['transformer', 'bert']:
                        optimizer = getattr(torch.optim, args.optimizer)([{'params': params, 'lr': optimizer.param_groups[0]['lr'] / 2.0}, 
                                                                      {'params': encoder_params, 'lr': optimizer.param_groups[1]['lr'] / 2.0},
                                                                      {'params': attn_params, 'lr': optimizer.param_groups[2]['lr'] / 2.0}])
                    else: # r2d2-based
                        optimizer = getattr(torch.optim, args.optimizer)([{'params': other_params, 'lr': optimizer.param_groups[0]['lr'] / 2.0},
                                                        {'params': r2d2_params, 'lr': optimizer.param_groups[1]['lr'] / 2.0},
                                                        {'params': parser_params, 'lr': optimizer.param_groups[2]['lr'] / 2.0}])
                # save checkpoint
                torch.save({
                    'model': model.state_dict(),
                    'best_model': best_model,
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'lr_controller': lr_controller,
                    'scaler': scaler.state_dict(),
                    'cuda_rng_state': torch.cuda.random.get_rng_state(),
                }, ckpt_path)
                # pre-terminate to avoid saving problem
                if (time.time() - args.start_time) >= args.time_limit:
                    logger.info('Training time is almost up -- terminating.')
                    exit(0)
        model.eval()
        with torch.no_grad():
            curr_f1, val_loss, val_model_loss_step = validate(args.task, data_loader['development'], model, getloss=True, use_argmax=args.use_argmax)
            
        model.train()

    # finished training, testing
    assert best_model is not None
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        test_f1 = validate(args.task, data_loader['test'], model, use_argmax=args.use_argmax)
    logger.info(f'Test F1 {test_f1 * 100:6.2f}%')


if __name__ == '__main__':
    main()
