from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from Multi_VAE.dataset import Multi_VAE_DataLoader


class MultiVaeModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        loader = Multi_VAE_DataLoader(data_dir)
        self.train_data = loader.load_data("train")
        self.vad_data_tr, self.vad_data_te = loader.load_data("validation")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=TrainDataset(self.train_data),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=ValidDataset(self.vad_data_tr, self.vad_data_te),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class TrainDataset(Dataset):
    def __init__(self, DATA):
        self.DATA = DATA
        self.N = DATA.shape[0]

    def naive_sparse2tensor(self, data):
        return torch.FloatTensor(data.toarray()).squeeze()

    def __len__(self):
        return self.DATA.shape[0]

    def __getitem__(self, idx):
        data = self.DATA[idx]
        data = self.naive_sparse2tensor(data)
        return data


class ValidDataset(Dataset):
    def __init__(self, data_tr, data_te):
        self.DATA = data_tr
        self.heldout_data = data_te

    def naive_sparse2tensor(self, data):
        return torch.FloatTensor(data.toarray()).squeeze()

    def __len__(self):
        return self.DATA.shape[0]

    def __getitem__(self, idx):
        data = self.DATA[idx]
        data = self.naive_sparse2tensor(data)

        data2 = self.heldout_data[idx]
        data2 = self.naive_sparse2tensor(data2)
        return (data, data2)


if __name__ == "__main__":
    _ = FMDataModule()
