import torch
import h5py
import time
import os
import gc
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

sorted_by_label = True # Whether the dataset should be sorted by label
chunked_storage = True # Whether the dataset should be stored in chunks, if sorted the the chunks are based on the labels
h5py_format = False # Whether the dataset should be stored using h5py, otherwise will be stored using pytorch
train_samples_per_class = 1024  # Number of samples per class for training
num_augmentations = 200 # Number of augmentations per image
val_samples_per_class = 64     # Number of samples per class for validation
output_dir = "./mnist_subset/chunks" if chunked_storage else "./mnist_subset"  # Directory to save the datasets
os.makedirs(output_dir, exist_ok=True)

device = "cpu"
print("Using: ", device)

augmentation_transformations = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.75, 1.25)),  # Rotate, translate and scale
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.025),  # Add Gaussian noise
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

def filter_indices_by_class(dataset):
    start_time = time.time()
    labels = torch.tensor([label for _, label in dataset])
    class_indices = {label: (labels == label).nonzero(as_tuple=True)[0].tolist() for label in range(10)}
    end_time = time.time()
    print(f"Filtered indices in: {end_time - start_time:.6f} seconds\n")
    return class_indices

def create_subset_indices(class_indices, train_samples_per_class, val_samples_per_class):
    start_time = time.time()
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        train_indices.extend(indices[:train_samples_per_class])
        val_indices.extend(indices[train_samples_per_class:train_samples_per_class + val_samples_per_class])
    end_time = time.time()
    print(f"Created subset indices in: {end_time - start_time:.6f} seconds\n")
    return train_indices, val_indices

def create_subsets(indices):
    start_time = time.time()
    images = torch.stack([mnist_train[i][0] for i in indices]).to(device)
    labels = torch.tensor([mnist_train[i][1] for i in indices]).to(device)
    if not sorted_by_label:
        print("Creating random permutation of indices")
        perm = torch.randperm(len(labels))
        images = images[perm]
        labels = labels[perm]
    end_time = time.time()
    print(f"Created subsets in: {end_time - start_time:.6f} seconds\n")
    return TensorDataset(images, labels)

def create_augmentations(images):
    aug1 = augmentation_transformations(images)
    aug2 = augmentation_transformations(images)
    return torch.stack([aug1, aug2])

def create_augmentation_chunk(image_chunk, label_chunk, n_augmentations, id):
    start_time_batch = time.time()
    image_chunk = image_chunk.to(device)
    label_chunk = label_chunk.to(device)
    augmented_images = []
    labels = []
    print("Shape of images in chunk: ", image_chunk.shape)
    for _ in range(n_augmentations // 2):
        augmented = torch.stack([create_augmentations(img) for img in image_chunk])
        augmented_images.append(augmented)
        labels.append(label_chunk)
    augmented_images = torch.cat(augmented_images)
    labels = torch.cat(labels)
    end_time_batch = time.time()
    print(f"Finished chunk: {id} in {end_time_batch - start_time_batch:.6f} seconds")
    return augmented_images, labels

def create_augmented_dataset(dataset, n_augmentations, batch_size):
    start_time = time.time()
    print("Started augmentation generation")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    augmented_images = []
    labels = []
    print("Amount of batches: ", len(dataloader))
    batch_counter = 0
    for batch_images, batch_labels in dataloader:
        aug_images, aug_labels = create_augmentation_chunk(batch_images, batch_labels, n_augmentations, batch_counter)
        augmented_images.append(aug_images)
        labels.append(aug_labels)
        batch_counter += 1
    augmented_images = torch.cat(augmented_images).to(device)  # Shape: [num_samples, 2, 1, 28, 28]
    print("Shape of augmented images: ", aug_images.shape)
    labels = torch.cat(labels).to(device)
    end_time = time.time()
    print(f"Created augmentations in: {end_time - start_time:.6f} seconds\n")
    return augmented_images, labels

def store_h5py(images, labels, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip')
        f.create_dataset('labels', data=labels)
    print(f"Stored in h5 format under: {file_path}\n") 

def store_torch(images, labels, file_path):
    torch.save((images, labels), file_path)
    print(f"Stored in pt format under: {file_path}\n") 

def store_full_augmented_dataset(dataset, train_set):
    prefix = "train" if train_set else "val"
    suffix = ".h5" if h5py_format else ".pt"
    num_aug = num_augmentations if train_set else 10
    augmented, labels = create_augmented_dataset(dataset, num_aug, train_samples_per_class)
    file_path = os.path.join(output_dir, f"mnist_{prefix}_subset_{train_samples_per_class}_per_class_aug_{num_augmentations}{suffix}")
    store_h5py(augmented, labels, file_path) if h5py_format else store_torch(augmented, labels, file_path)
    print(f"Augmented {prefix} subset created with shape: {augmented.shape}")
    print(f"Saved to {output_dir}\n")
    del augmented
    del labels
    gc.collect()

def store_chunked_train_augmented_dataset(dataset, batch_size):
    start_time = time.time()
    suffix = ".h5" if h5py_format else ".pt"
    print("Started chunked augmentation generation")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunk_counter = 0
    for image_chunk, label_chunk in dataloader:
        augmented, labels = create_augmentation_chunk(image_chunk, label_chunk, num_augmentations, chunk_counter)
        file_path = os.path.join(output_dir, f"mnist_train_subset_{train_samples_per_class}_per_class_aug_{num_augmentations}_chunk_{chunk_counter}{suffix}")
        store_h5py(augmented, labels, file_path) if h5py_format else store_torch(augmented, labels, file_path)
        print(f"Augmented training subset chunk created with shape: {augmented.shape}")
        print(f"Saved to {output_dir}\n")
        del augmented
        del labels
        gc.collect()
        chunk_counter += 1 
    end_time = time.time()
    print(f"Stored all chunks in: {end_time - start_time:.6f} seconds\n")

transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root=".", train=True, download=True, transform=transform)

class_indices = filter_indices_by_class(mnist_train)
train_indices, val_indices = create_subset_indices(class_indices, train_samples_per_class, val_samples_per_class)

train = create_subsets(train_indices)
val = create_subsets(val_indices)

print("Training and validation subsets created")
print(f"Training subset length {len(train)}")
print(f"Validation subset length: {len(val)}\n")

del mnist_train
gc.collect()

if chunked_storage:
    store_chunked_train_augmented_dataset(train, train_samples_per_class)    
else:
    store_full_augmented_dataset(train, train_set=True)

del train
gc.collect()

store_full_augmented_dataset(val, train_set=False)