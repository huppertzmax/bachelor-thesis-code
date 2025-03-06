import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from training.utils.utils import log_msg

class TinyMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset/mnist_subset/",
        train_subset_name: str = "mnist_train_subset_1024_per_class.pt",
        val_subset_name: str = "mnist_val_subset_256_per_class.pt",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_subset_name = train_subset_name
        self.val_subset_name = val_subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.augmentations = TinyAugmentations()
        self.test_augmentations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        self.train_dataset = torch.load(self.data_dir + self.train_subset_name)
        self.val_dataset = torch.load(self.data_dir + self.val_subset_name)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TensorDataset(*self.train_dataset)
            self.train_dataset = AugmentedDataset(self.train_dataset, transform=self.augmentations)
            self.val_dataset = TensorDataset(*self.val_dataset)
            self.val_dataset = AugmentedDataset(self.val_dataset, transform=self.augmentations)

        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=self.test_augmentations)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class TinyAugmentations():
    def __init__(self):
        self.augmentation_transformations = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.75, 1.25)),  # Rotate, translate and scale
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.025),  # Add Gaussian noise
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize
        ])

    def __call__(self, sample):
        return self.augmentation_transformations(sample.float())
    
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image1 = self.transform(image.float())
        image2 = self.transform(image.float())
        return image1, image2, label