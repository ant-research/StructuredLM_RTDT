import argparse
import torch
import codecs
import traceback
import json
import os
from eval.tree_file_wrapper import TreeFileWrapper
from model.fast_r2d2_dp_classification import FastR2D2DPClassification
from eval.r2d2_wrapper import R2D2dpParserWrapper
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--model_path', required=True, type=str)
    cmd.add_argument('--parser_only', required=False, action='store_true')
    cmd.add_argument('--enable_cuda_extension', default=False, action='store_true')
    cmd.add_argument('--config_path', required=True, type=str)
    cmd.add_argument('--corpus_path', required=True, type=str)
    cmd.add_argument('--in_word', action='store_true', default=False)
    cmd.add_argument('--output_path', default='./pred_trees.txt', type=str)
    cmd.add_argument('--transformer_parser', default=False, action='store_true')
    cmd.add_argument('--to_latex_tree', action='store_true', default=False)
    cmd.add_argument('--tree_input', default=None, type=str)
    cmd.add_argument("--enable_topdown", default=False, action='store_true')
    cmd.add_argument("--enable_exclusive", default=False, action='store_true')
    cmd.add_argument("--label_num", required=True, type=int)
    cmd.add_argument("--tsv_column", default=-1, type=int)
    cmd.add_argument("--atis", default=-1, type=int)
    options = cmd.parse_args()
    model_cls = FastR2D2DPClassification #R2D2Cuda
    labels = [str(label_idx) for label_idx in range(options.label_num)]

    if options.tsv_column != -1:
        data = pd.read_csv(options.corpus_path, sep='\t', header=0)
        lines = []
        for row in data.values:
            lines.append(row[options.tsv_column])

    elif options.atis != -1:
        lines = []
        labels = []
        with codecs.open(os.path.join(options.corpus_path, 'standard_format/rasa/train.json'), mode='r', encoding='utf-8') as f:
            data = json.load(f)
            for line_json in data['rasa_nlu_data']['common_examples']:
                if '+' in line_json['intent']:
                    lines.append((line_json['text'],line_json['intent']))
        with open(os.path.join(options.corpus_path, 'raw_data/ms-cntk-atis/atis.dict.intent.csv'),'r') as f:
                for intent in f:
                    if '+' not in intent:
                        labels.append(intent.strip())
    else:
        lines = []
        with codecs.open(options.corpus_path, mode='r', encoding='utf-8') as f:
            for _line in f:
                lines.append(_line) 
    
    if options.tree_input is None:
        predictor = R2D2dpParserWrapper(config_path=options.config_path, model_cls=model_cls, 
                                    parser_only=options.parser_only,
                                    model_path=options.model_path,
                                    labels=labels,
                                    enable_topdown=options.enable_topdown,
                                    enable_exclusive=options.enable_exclusive,
                                    sep_word=' ', in_word=options.in_word, device=device)
    else:
        predictor = TreeFileWrapper(options.tree_input)

    with codecs.open(os.path.join(options.output_path), mode='w', encoding='utf-8') as out:
        count = 0
        for l in lines:
            label = None
            if isinstance(l,tuple):
                l, label = l
            tokens = l.split()
            if len(tokens) >= 2:
                if not options.to_latex_tree:
                    tree = predictor.print_binary_ptb(tokens)
                    print(tree, file=out)
                else:
                    tree = predictor.print_latex_tree(tokens, label)
                    print(tree, file=out)