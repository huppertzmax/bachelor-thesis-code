import os
import torch
import random
import json
from argparse import ArgumentParser
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_cka import CKA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils import load_tiny_mnist_backbone, tensor_to_json_compatible

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = ArgumentParser()

parser.add_argument("--model1_name", type=str)
parser.add_argument("--model2_name", type=str)
parser.add_argument("--model1_path", type=str)
parser.add_argument("--model2_path", type=str)
parser.add_argument("--path_suffix", type=str)

args = parser.parse_args()
print(args)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

model1_name = args.model1_name
model2_name = args.model2_name
model1_ckpt=  args.model1_path
model2_ckpt = args.model2_path
model1 = load_tiny_mnist_backbone(model1_ckpt) 
model2 = load_tiny_mnist_backbone(model2_ckpt)

layers_model1 = [name for name, module in model1.named_modules()]
layers_model2 = [name for name, module in model2.named_modules()]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device\n')

cka = CKA(model1, model2,
          model1_name=model1_name,   
          model2_name=model2_name,  
          model1_layers=layers_model1,
          model2_layers=layers_model1, 
          device=device)

cka.compare(dataloader) 

results = cka.export()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/cka", f"{timestamp}_cka_{args.path_suffix}")
os.makedirs(folder_name, exist_ok=True)

similarity_matrix = results['CKA'].numpy()

results = tensor_to_json_compatible(results)
results[model1_name] = model1_ckpt
results[model2_name] = model2_ckpt

with open(folder_name + "/cka_results.json", 'w') as f:
    json.dump(results, f, indent=4)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=layers_model2, yticklabels=layers_model1, cmap='magma')
plt.title('CKA Similarity Heatmap')
plt.xlabel(model2_name)
plt.gca().invert_yaxis()
plt.ylabel(model1_name)
plt.tight_layout()

plt.savefig(folder_name + "/cka_sns_heatmap.png")
plt.show()