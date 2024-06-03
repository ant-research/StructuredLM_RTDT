from utils.generation_util import GPTGenerationUtil, GenerationUtil
from utils.beam_searcher import R2D2GenFastBeamSearcher


def create_generator(model_type, model, device, gpt_config, beam_size=2, sampling=True, word_sync=True):
    if model_type == "gpt": 
        return GPTGenerationUtil(model, device)
    elif model_type == "r2d2-gen-fast":
        if word_sync:
            print("word_sync !!!!!!")
            return R2D2GenFastBeamSearcher(model, gpt_config, device, beam_size=beam_size, sampling=sampling)
        else:
            print("not word_sync !!!!!!")
            return GenerationUtil(model, device, gpt_config)
    elif model_type == "r2d2-gen":
        raise Exception('current not suppport r2d2-gen')
    else:
        raise Exception('current not suppport model_type')


if __name__ == "__main__":
    pass