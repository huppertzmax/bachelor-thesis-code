import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split

from training.utils.utils import log_msg

class KFoldDataModule(pl.LightningDataModule):
    def __init__(self, dataset, k=5, batch_size=64, val_split=0.1, train_majority=True):
        super().__init__()
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_majority = train_majority
        self.kfold = KFold(n_splits=k)
        self.fold_indices = list(self.kfold.split(range(len(dataset))))
        self.fold_index = None

    def setup(self, stage=None):
        train_indices, test_indices = self.fold_indices[self.fold_index]

        if self.train_majority:
            train_indices, test_indices = self.fold_indices[self.fold_index]
        else:
            test_indices, train_indices = self.fold_indices[self.fold_index]

        train_size = int(len(train_indices) * (1 - self.val_split))
        val_size = len(train_indices) - train_size
        train_subset, val_subset = random_split(
            Subset(self.dataset, train_indices),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = Subset(self.dataset, test_indices)

        log_msg(f"Length of train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)} in fold {self.fold_index}")

    def update_fold_index(self, fold_index):
        self.fold_index = fold_index

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
