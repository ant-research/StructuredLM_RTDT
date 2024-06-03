import torch

def expand_embedding(model_path, output_path, keys, org_dim, target_dim):
    state_dicts = torch.load(model_path, map_location=lambda a, b: a)
    new_state_dicts = {}
    for key, val in state_dicts.items():
        hit_key = False
        for target_key in keys:
            if key.find(target_key) >= 0:
                assert val.shape[0] == org_dim
                hit_key = True
                new_val = torch.zeros(target_dim, val.shape[1], dtype=val.dtype)
                new_val.data.normal_(mean=0, std=0.02)
                new_val[:org_dim, :] = val
                new_state_dicts[key] = new_val
                break
        if not hit_key:
            new_state_dicts[key] = val
    torch.save(new_state_dicts, output_path)

def convert_to_gpt_pretrained_model(model_path, output_path):
    state_dicts = torch.load(model_path, map_location=lambda a, b: a)
    out_dict = {}
    for key, val in state_dicts.items():
        new_key = key.replace('module.gpt.', '')
        out_dict[new_key] = val

    torch.save(out_dict, output_path)

def print_specific_dim(model_path, keys):
    state_dicts = torch.load(model_path, map_location=lambda a, b: a)
    for key, val in state_dicts.items():
        for target_key in keys:
            if key.find(target_key) >= 0:
                print("ori_key_value: ", key, val.shape[0])
                break

def print_dim(model_path):
    state_dicts = torch.load(model_path, map_location=lambda a, b: a)
    for key, val in state_dicts.items():
        print("ori_key_value: ", key, val.shape)

def extract_xsumwarpper(model_path, output_path):
    state_dicts = torch.load(model_path, map_location=lambda a, b: a)
    out_dict = {}
    for key, val in state_dicts.items():
        new_key = key.replace('model.', '')
        out_dict[new_key] = val

    torch.save(out_dict, output_path)

