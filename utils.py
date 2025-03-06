import torch
from torchvision import models
from training.models.tiny_mnist_backbone import TinyMNISTBackbone

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tensor_to_json_compatible(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_json_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_json_compatible(item) for item in obj]
    else:
        return obj
    
def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]
    model = models.resnet18()  
    model.load_state_dict(modify_state_dict(state_dict))  
    model.eval()
    return model  

def load_tiny_mnist_backbone(ckpt_path):
    backbone = TinyMNISTBackbone()
    backbone.load_state_dict(update_ckpt_dict(ckpt_path))
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)

def update_ckpt_dict(path):
    checkpoint = torch.load(path, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    print(f"Original state dict keys: {state_dict.keys()}")
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "")  
            updated_state_dict[new_key] = value
    
    print(f"Updated state dict keys: {updated_state_dict.keys()}")    
    return updated_state_dict

def modify_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "")  
            new_state_dict[new_key] = value
    return new_state_dict