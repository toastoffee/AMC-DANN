import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rml_dataset import RmlHelper


class DataloaderHelper:

    @staticmethod
    def initialize_random_seed():
        seed = 2016
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def dataloader_10a(batch_size: int,
                       train_ratio: float = 0.8):

        DataloaderHelper.initialize_random_seed()
        rml201610a = RmlHelper.rml201610a()

        n_examples = rml201610a.sample_num
        n_train = int(train_ratio * n_examples)
        n_valid = n_examples - n_train
        lengths = [n_train, n_valid]

        train_subset, valid_subset = torch.utils.data.random_split(rml201610a, lengths)

        train_loader_all = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        valid_loader_all = DataLoader(dataset=valid_subset, batch_size=batch_size)

        return train_loader_all, valid_loader_all

    @staticmethod
    def dataloader_04c(batch_size: int,
                       train_ratio: float = 0.8):

        DataloaderHelper.initialize_random_seed()
        rml201604a = RmlHelper.rml201604c()

        n_examples = rml201604a.sample_num
        n_train = int(train_ratio * n_examples)
        n_valid = n_examples - n_train
        lengths = [n_train, n_valid]

        train_subset, valid_subset = torch.utils.data.random_split(rml201604a, lengths)

        train_loader_all = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        valid_loader_all = DataLoader(dataset=valid_subset, batch_size=batch_size)

        return train_loader_all, valid_loader_all


if __name__ == "__main__":
    train, valid = DataloaderHelper.dataloader_10a(64)
