import argparse
import torch
import codecs
import traceback
import os
from model.r2d2_cuda import R2D2Cuda
from eval.r2d2_wrapper import R2D2ParserWrapper


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--model_path', required=True, type=str)
    cmd.add_argument('--parser_path', required=False, type=str)
    cmd.add_argument('--parser_only', required=False, action='store_true')
    cmd.add_argument('--enable_cuda_extension', default=False, action='store_true')
    cmd.add_argument('--config_path', required=True, type=str)
    cmd.add_argument('--corpus_path', required=True, type=str)
    cmd.add_argument('--in_word', action='store_true', default=False)
    cmd.add_argument('--output_path', default='./pred_trees.txt', type=str)
    options = cmd.parse_args()
    model_cls = R2D2Cuda
    
    predictor = R2D2ParserWrapper(config_path=options.config_path, model_cls=model_cls, 
                                  parser_only=options.parser_only,
                                  model_path=options.model_path, 
                                  parser_path=options.parser_path,
                                  sep_word=' ', in_word=options.in_word, device=device)

    with codecs.open(options.corpus_path, mode='r', encoding='utf-8') as f, \
            codecs.open(os.path.join(options.output_path), mode='w', encoding='utf-8') as out:
        for l in f:
            try:
                tokens = l.split()
                if len(tokens) >= 2:
                    tree = predictor.print_binary_ptb(tokens)
                    print(tree, file=out)
            except:
                traceback.print_exc()
                print(l)
                continue