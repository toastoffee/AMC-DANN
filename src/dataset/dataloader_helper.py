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
    def dataloader_train_valid_split(dataset: Dataset,
                                     batch_size: int,
                                     train_ratio: float = 0.6):
        DataloaderHelper.initialize_random_seed()

        n_examples = dataset.sample_num
        n_train = int(train_ratio * n_examples)
        n_valid = n_examples - n_train
        lengths = [n_train, n_valid]

        train_subset, valid_subset = torch.utils.data.random_split(dataset, lengths)

        train_loader_all = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        valid_loader_all = DataLoader(dataset=valid_subset, batch_size=batch_size)

        return train_loader_all, valid_loader_all

    @staticmethod
    def dataloader_10a(batch_size: int,
                       train_ratio: float = 0.6):

        train_loader_all, valid_loader_all = DataloaderHelper.dataloader_train_valid_split(
            RmlHelper.rml201610a(), batch_size, train_ratio
        )

        return train_loader_all, valid_loader_all

    @staticmethod
    def dataloader_04c(batch_size: int,
                       train_ratio: float = 0.6):

        train_loader_all, valid_loader_all = DataloaderHelper.dataloader_train_valid_split(
            RmlHelper.rml201604c(), batch_size, train_ratio
        )

        return train_loader_all, valid_loader_all

    @staticmethod
    def dataloader_22(batch_size: int,
                      train_ratio: float = 0.6):

        train_loader_all, valid_loader_all = DataloaderHelper.dataloader_train_valid_split(
            RmlHelper.rml22(), batch_size, train_ratio
        )

        return train_loader_all, valid_loader_all


if __name__ == "__main__":
    train, valid = DataloaderHelper.dataloader_10a(64)
