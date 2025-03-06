import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl

class MultiFileLazyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.file_data = [None] * len(file_paths)  
        self.index_map = []

        for file_idx, file_path in enumerate(file_paths):
            images, _ = torch.load(file_path)
            self.index_map.extend([(file_idx, i) for i in range(len(images))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]

        if self.file_data[file_idx] is None:
            self.file_data[file_idx] = torch.load(self.file_paths[file_idx])

        images, labels = self.file_data[file_idx]
        return images[sample_idx], labels[sample_idx]

class TinyChunkedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, train_subset, val_subset, num_workers=4,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.train_subset = train_subset
        self.val_subset = val_subset, 
        self.file_paths = [f"{self.data_dir}chunks/{self.train_subset}{i}.pt" for i in range(10)]
        print("Using chunks of those files: ", self.file_paths)

    def setup(self, stage=None):
        self.train_dataset = MultiFileLazyDataset(self.file_paths)
        file_path = f"{self.data_dir}chunks/{self.val_subset[0]}"
        print(file_path)
        self.val_dataset = torch.load(file_path)
        self.val_dataset = TensorDataset(*self.val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )