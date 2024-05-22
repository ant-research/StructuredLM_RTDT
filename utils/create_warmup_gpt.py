from model.model_factory import create_model
import torch
import re


def create_warmup_gpt(r2d2_config_path, gpt_config_path, r2d2_gpt_model_path, output_gpt_path):
    r2d2_gen = create_model('r2d2-gen', r2d2_config_path, gpt_config_path)
    r2d2_gen.from_pretrain(r2d2_gpt_model_path)

    gpt_model = create_model('gpt', None, gpt_config_path)

    gpt_model.wte = r2d2_gen.embeddings

    torch.save(gpt_model.gpt.state_dict(), output_gpt_path)

def create_warmup_r2d2_gen_fast(r2d2_config_path, gpt_config_path, r2d2_gpt_model_path, output_path):
    r2d2_gen = create_model('r2d2-gen', r2d2_config_path, gpt_config_path)
    r2d2_gen.from_pretrain(r2d2_gpt_model_path)

    r2d2_gen_fast = create_model('r2d2-gen-fast', r2d2_config_path, gpt_config_path)
    r2d2_gen_fast.r2d2 = r2d2_gen.r2d2
    r2d2_gen_fast.embeddings.weight.data = r2d2_gen.embeddings.weight.data
    r2d2_gen_fast.insideoutside_dense = r2d2_gen.insideoutside_dense
    torch.save(r2d2_gen_fast.state_dict(), output_path)

def create_r2d2_gen_fast_from_gpt(r2d2_config_path, gpt_config_path, gpt_model_path, output_path):
    r2d2_gen_fast = create_model('r2d2-fast', r2d2_config_path, gpt_config_path)
    gpt_model_dict = torch.load(gpt_model_path)
    wte_weights = gpt_model_dict['wte.weight']
    vocab_sz, _ = wte_weights.shape
    # r2d2_gen_fast.embeddings = wte_weights
    r2d2_gen_fast.embeddings.weight.data[:vocab_sz, :] = wte_weights
    torch.save(r2d2_gen_fast.state_dict(), output_path)

def create_gpt_from_gpt(gpt_config_path, gpt_model_path, output_path):
    model = create_model('gpt', None, gpt_config_path)
    gpt_model_dict = torch.load(gpt_model_path)
    wte_weights = gpt_model_dict['wte.weight']
    vocab_sz, _ = wte_weights.shape
    model.gpt.transformer.wte.weight.data[:vocab_sz, :] = wte_weights
    wpe_weights = gpt_model_dict['wpe.weight']
    wpe_sz, _ = wpe_weights.shape
    model.gpt.transformer.wpe.weight.data[:wpe_sz, :] = wpe_weights
    torch.save(model.gpt.state_dict(), output_path)

def merge_r2d2_gen_fast_with_gpt(r2d2_config_path, gpt_config_path, r2d2_gen_model, gpt_model_path, output_path):
    from transformers import AutoConfig
    r2d2_gen_fast = create_model('r2d2-gen-fast', r2d2_config_path, gpt_config_path)
    r2d2_gen_fast.from_pretrain(r2d2_gen_model, strict=False)
    gpt_config = AutoConfig.from_pretrained(gpt_config_path)
    gpt_model_dict = torch.load(gpt_model_path)
    action_layer_num = len(r2d2_gen_fast.action_layers.h)
    # layer pattern
    # prefix h.[0-9]+.*
    # h.*.ln_1.weight
    # h.0.ln_1.bias
    # h.0.attn.bias
    # h.0.attn.c_attn.weight
    # h.0.attn.c_attn.bias
    # h.0.attn.c_proj.weight
    # h.0.attn.c_proj.bias
    # h.0.ln_2.weight
    # h.0.ln_2.bias
    # h.0.mlp.c_fc.weight
    # h.0.mlp.c_fc.bias
    # h.0.mlp.c_proj.weight
    # h.0.mlp.c_proj.bias

    # layer_norm:
    # ln_f.weight
    # ln_f.bias

    # embeddings:
    # wte.weight
    # wpe.weight

    # classify parameters
    layer_pattern = r'h.[0-9]+.'
    norm_pattern = r'ln_f'
    embedding_pattern = r'(wte|wpe).weight'

    layer_parameters = {}
    embedding_parameters = {}
    norm_layer_parameters = {}
    for k, v in gpt_model_dict.items():
        if re.match(layer_pattern, k):
            layer_parameters[k] = v
        elif re.match(norm_pattern, k):
            norm_layer_parameters[k] = v
        elif re.match(embedding_pattern, k):
            embedding_parameters[k] = v
        else:
            print(f'{k} match nothing')

    wte_weights = gpt_model_dict['wte.weight']
    vocab_sz, _ = wte_weights.shape
    r2d2_gen_fast.embeddings.weight.data[:vocab_sz, :] = wte_weights
    wpe_weights = gpt_model_dict['wpe.weight']
    wpe_sz, _ = wpe_weights.shape
    r2d2_gen_fast.action_layers.wpe.weight.data[:wpe_sz, :] = wpe_weights

    r2d2_gen_fast.generation_layers.ln_f.weight.data = gpt_model_dict['ln_f.weight']
    r2d2_gen_fast.generation_layers.ln_f.bias.data = gpt_model_dict['ln_f.bias']
    r2d2_gen_fast.bos_embedding.data = gpt_model_dict['wte.weight'][gpt_config.eos_token_id, :]

    layer_extract_pattern = r'h.(\d+).*'
    for k, v in layer_parameters.items():
        # extract layer_num
        match = re.search(layer_extract_pattern, k)
        if match:
            layer_num = int(match.group(1))
            tgt = r2d2_gen_fast.action_layers
            if layer_num >= action_layer_num:
                tgt = r2d2_gen_fast.generation_layers
                k = k.replace(f'.{str(layer_num)}.', f'[{str(layer_num - action_layer_num)}].')
            else:
                k = k.replace(f'.{str(layer_num)}.', f'[{str(layer_num)}].')

            param = eval(f'tgt.{k}')
            param.data = v
    torch.save(r2d2_gen_fast.state_dict(), output_path)
