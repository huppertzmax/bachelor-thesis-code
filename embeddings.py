import torch
import os  
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser

from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from training.utils.ckpt_loading import update_ckpt_dict

def load_backbone(ckpt_path, device): 
    start_time = time.time()
    print(f"Start loading model from checkpoint: {ckpt_path}")
    backbone = TinyMNISTBackbone()
    backbone.load_state_dict(update_ckpt_dict(ckpt_path))
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    end_time = time.time()
    print(f"Loaded model in: {end_time - start_time:.6f} seconds\n")
    return backbone

def load_dataset(data_path):
    start_time = time.time()
    print(f"Start loading dataset from path: {data_path}")
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    end_time = time.time()
    print(f"Loaded dataset in: {end_time - start_time:.6f} seconds\n")
    return dataset

def store_embedding(embedding, storage_path, num_samples_per_class, num_augmentations, chunked=False, chunk_nr=0):
    os.makedirs(storage_path, exist_ok=True)
    if chunked:
        file_path = storage_path + f"/embedding_{num_samples_per_class}_{num_augmentations}_chunk_{chunk_nr}.pt"
    else:
        file_path = storage_path + f"/embedding_{num_samples_per_class}_{num_augmentations}.pt"
    torch.save(embedding, file_path)
    chunked_note = "chunk" if chunked else ""
    print(f"Stored embedding {chunked_note} under {file_path}\n")

def store_embedding_numpy(embedding, storage_path, num_samples_per_class, num_augmentations):
    os.makedirs(storage_path, exist_ok=True)
    file_path = storage_path + f"/embedding_{num_samples_per_class}_{num_augmentations}"
    np.save(file_path, embedding.cpu().numpy())

def calculate_chunk_embedding(num_augmentations, num_samples_per_class, ckpt_path, data_path, storage_path, num_chunks, device):
    backbone = load_backbone(ckpt_path, device)
    full_embedding = []

    start_time_full = time.time()
    print(f"Start calculating neural network embedding in {num_chunks} chunks")
    for chunk in range(num_chunks): 
        data_path_chunk = data_path + "_chunk_" +str(chunk) + ".pt"
        dataset = load_dataset(data_path_chunk)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        start_time = time.time()
        print(f"Start calculating neural network embedding of chunk {chunk} in {len(loader)} batches")
        embeddings = []
        batch_counter = 0
        for batch in loader:
            images, y = batch 
            images = images.to(device)
            images = images.view(images.size(dim=0)*2, 1, 28, 28)
            embedding = backbone(images)
            if len(loader) >= 100 and (batch_counter+1) % 50 == 0:  
                print(f"Shape of embedding of batch {batch_counter} is: {embedding.shape}")
            if False in y.eq(torch.full(y.shape, chunk)):
                print("Labels are: ", y)
                print("but should be: ", torch.full(y.shape, chunk))
                exit()
            batch_counter += 1
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)
        full_embedding.append(embeddings)
        print(f"\nShape of produced embedding: {embeddings.shape}")
        end_time = time.time()
        print(f"Calculated embedding of chunk {chunk} in: {end_time - start_time:.6f} seconds\n")

    full_embedding = torch.cat(full_embedding)
    print(f"\nShape of full produced embedding: {full_embedding.shape}")
    end_time_full = time.time()
    print(f"Calculated full embedding in: {end_time_full - start_time_full:.6f} seconds\n")
    store_embedding_numpy(full_embedding, storage_path, num_samples_per_class, num_augmentations)

def calculate_embedding(num_augmentations, num_samples_per_class, ckpt_path, data_path, storage_path, device):
    backbone = load_backbone(ckpt_path, device)

    dataset = load_dataset(data_path)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    start_time = time.time()
    print(f"Start calculating neural network embedding in {len(loader)} batches")
    embeddings = []
    batch_counter = 0
    for batch in loader:
        images, _ = batch 
        images = images.to(device)
        images = images.view(images.size(dim=0)*2, 1, 28, 28)
        embedding = backbone(images)
        print(f"Shape of embedding of batch {batch_counter} is: {embedding.shape}")
        batch_counter += 1
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings)
    print(f"\nShape of produced embedding: {embeddings.shape}")
    end_time = time.time()
    print(f"Calculated embedding in: {end_time - start_time:.6f} seconds\n")
    store_embedding(embeddings, storage_path, num_samples_per_class, num_augmentations)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/mnist_subset/")
    parser.add_argument("--ckpt_path", type=str, default="results/polar-bush-109/last.ckpt")
    parser.add_argument("--ckpt_name", type=str, default="polar-bush-109")
    parser.add_argument("--num_samples_per_class", type=int, default=1024)
    parser.add_argument("--num_augmentations", type=str, default=200)
    parser.add_argument("--chunked_data", type=bool, default=False)
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--storage_path", type=str, default="./results/embeddings/")

    args = parser.parse_args()
    
    suffix = "/chunks" if args.chunked_data else ""
    args.storage_path = args.storage_path + args.ckpt_name + suffix

    if args.chunked_data: 
        args.data_path = f"{args.data_path}chunks/mnist_train_subset_{args.num_samples_per_class}_per_class_aug_{args.num_augmentations}"
        calculate_chunk_embedding(args.num_augmentations, args.num_samples_per_class, args.ckpt_path, args.data_path, args.storage_path, args.num_chunks, device=device)
    else: 
        args.data_path = f"{args.data_path}mnist_train_subset_{args.num_samples_per_class}_per_class_aug_{args.num_augmentations}.pt"
        calculate_embedding(args.num_augmentations, args.num_samples_per_class, args.ckpt_path, args.data_path, args.storage_path, device=device)
