import torch

from training.utils.utils import log_msg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def update_ckpt_dict(path):
    checkpoint = torch.load(path, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    log_msg(f"Original state dict keys: {state_dict.keys()}")
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "")  
            updated_state_dict[new_key] = value
    
    log_msg(f"Updated state dict keys: {updated_state_dict.keys()}")    
    return updated_state_dict