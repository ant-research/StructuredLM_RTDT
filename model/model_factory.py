from model.r2d2_insideoutside import InsideOutsideModule
from transformers import AutoModel, AutoConfig
from model.gpt2_flash_attn import GPT2Model
from model.gpt2_wrapper import GPT2Wrapper
from model.glue_wrapper import GlueWrapper
from model.xsum_wrapper import XSumWrapper

def _create_generative_r2d2(model_cls, gpt_config_path, r2d2_config_path, gradient_checkpoint=False):
    gpt_config = AutoConfig.from_pretrained(gpt_config_path)
    vocab_size = gpt_config.vocab_size
    gpt = GPT2Model(gpt_config, no_embedding=True)
    gpt.gradient_checkpointing = gradient_checkpoint
    r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
    r2d2_config.vocab_size = vocab_size
    r2d2 = InsideOutsideModule(r2d2_config)
    
    r2d2_input_dim = r2d2.input_dim
    gpt_input_dim = gpt_config.n_embd
    generative_r2d2 = model_cls(r2d2, gpt, vocab_size, r2d2_input_dim, gpt_input_dim, \
                                ext_vocab_size=r2d2_config.ext_vocab_size)
    return generative_r2d2

def create_model(model_type, r2d2_config_path, gpt_config_path, fix_embeddings=False, gradient_checkpoint=False):
    if model_type == 'r2d2':
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        vocab_size = gpt_config.vocab_size
        r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
        r2d2_config.vocab_size = vocab_size
        r2d2 = InsideOutsideModule(r2d2_config)
        
        r2d2_input_dim = r2d2.input_dim
        gpt_input_dim = gpt_config.n_embd
        from model.generative_r2d2 import GenerativeR2D2
        generative_r2d2 = GenerativeR2D2(r2d2, None, vocab_size, r2d2_input_dim, gpt_input_dim, \
                                         ext_vocab_size=r2d2_config.ext_vocab_size)
        return generative_r2d2
    elif model_type == 'r2d2-fast':
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        vocab_size = gpt_config.vocab_size
        r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
        r2d2_config.vocab_size = vocab_size
        r2d2 = InsideOutsideModule(r2d2_config)
        
        r2d2_input_dim = r2d2.input_dim
        gpt_input_dim = gpt_config.n_embd
        from model.generative_r2d2_fast import FastGenerativeR2D2
        generative_r2d2 = FastGenerativeR2D2(r2d2, None, None, vocab_size, r2d2_input_dim, gpt_input_dim, \
                                             ext_vocab_size=r2d2_config.ext_vocab_size,
                                             fix_embeddings=fix_embeddings)
        return generative_r2d2
    elif model_type == 'r2d2-gen':
        from model.generative_r2d2 import GenerativeR2D2
        return _create_generative_r2d2(GenerativeR2D2, gpt_config_path, r2d2_config_path, 
                                       gradient_checkpoint=gradient_checkpoint)
    # elif model_type == 'r2d2-gen-ext':
    #     from model.generative_r2d2_ext import GenerativeR2D2
    #     return _create_generative_r2d2(GenerativeR2D2, gpt_config_path, r2d2_config_path, 
    #                                    gradient_checkpoint=gradient_checkpoint)
    elif model_type in ['r2d2-gen-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext']:
        if model_type == 'r2d2-gen-fast':
            from model.generative_r2d2_fast import FastGenerativeR2D2
        elif model_type == 'r2d2-gen-fast-struct':
            from model.generative_r2d2_fast_abl_structonly import FastGenerativeR2D2
        elif model_type == 'r2d2-gen-fast-ext':
            from model.generative_r2d2_fast_ext import FastGenerativeR2D2
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        vocab_size = gpt_config.vocab_size
        total_layer = gpt_config.n_layer
        gpt_config.n_layer = gpt_config.action_layer_num
        action_transformers = GPT2Model(gpt_config, no_embedding=True, no_layer_norm=True)
        action_transformers.gradient_checkpointing = gradient_checkpoint
        gpt_config.n_layer = total_layer - gpt_config.action_layer_num
        gpt_transformers = GPT2Model(gpt_config, no_embedding=True, no_extra_embedding=True)
        gpt_transformers.gradient_checkpointing = gradient_checkpoint
        r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
        r2d2_config.vocab_size = vocab_size
        r2d2 = InsideOutsideModule(r2d2_config)
        
        r2d2_input_dim = r2d2.input_dim
        gpt_input_dim = gpt_config.n_embd
        generative_r2d2 = FastGenerativeR2D2(r2d2, action_transformers, gpt_transformers, vocab_size, r2d2_input_dim,\
                                             gpt_input_dim, ext_vocab_size=r2d2_config.ext_vocab_size)
        return generative_r2d2
    elif model_type == 'gpt':
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        gpt = GPT2Wrapper(gpt_config)
        gpt.gpt.transformer.gradient_checkpointing = gradient_checkpoint
        return gpt
    elif model_type == 'llama':
        pass

def glue_create_model(model_type, r2d2_config_path, gpt_config_path, fix_embeddings=False, gradient_checkpoint=False, finetune_class_num=-1):
    if model_type == "r2d2-gen-fast" and finetune_class_num != -1:  # discriminant r2d2-gen-fast
        from model.generative_r2d2_fast import FastGenerativeR2D2_discriminant_glue
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        vocab_size = gpt_config.vocab_size
        total_layer = gpt_config.n_layer
        gpt_config.n_layer = gpt_config.action_layer_num
        action_transformers = GPT2Model(gpt_config, no_embedding=True, no_layer_norm=True)
        action_transformers.gradient_checkpointing = gradient_checkpoint
        gpt_config.n_layer = total_layer - gpt_config.action_layer_num
        gpt_transformers = GPT2Model(gpt_config, no_embedding=True, no_extra_embedding=True)
        r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
        r2d2_config.vocab_size = vocab_size
        r2d2 = InsideOutsideModule(r2d2_config)
        
        r2d2_input_dim = r2d2.input_dim
        gpt_input_dim = gpt_config.n_embd
        model = FastGenerativeR2D2_discriminant_glue(r2d2, action_transformers, gpt_transformers, vocab_size, r2d2_input_dim,\
                                             gpt_input_dim, ext_vocab_size=r2d2_config.ext_vocab_size)
    else:
        model = create_model(model_type, r2d2_config_path, gpt_config_path, fix_embeddings=fix_embeddings, gradient_checkpoint=gradient_checkpoint)
    glue_model = GlueWrapper(model, model_type, model.embedding_dim, finetune_class_num=finetune_class_num)
    return glue_model

def xsum_create_model(model_type, r2d2_config_path, gpt_config_path, fix_embeddings=False, gradient_checkpoint=False):
    model = create_model(model_type, r2d2_config_path, gpt_config_path, fix_embeddings=fix_embeddings, gradient_checkpoint=gradient_checkpoint)
    xsum_model = XSumWrapper(model)
    return xsum_model