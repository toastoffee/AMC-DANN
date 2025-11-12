import numpy as np
import torch
from torch.utils.data import Dataset


def set_seeds(seed: int):
    """
    set seed of pytorch and numpy
    :param seed: the random seed
    :return: void
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def dataset_random_split(dataset: Dataset,
                         lengths: list,
                         seed:    int):
    """
    wrapper of 'torch.utils.data.random_split' with random seeds setter
    :param dataset: Dataset
    :param lengths: can be ratios [0.6, 0.4] or numbers [100, 100]
    :param seed: random seed
    :return: random split datasets
    """
    set_seeds(seed)
    return torch.utils.data.random_split(dataset, lengths)


def random_subset(dataset: Dataset,
                  cnt:     int,
                  seed:    int):
    """
    randomly pick samples from dataset and return a subset
    :param dataset: dataset to select from
    :param cnt: the count of samples selected
    :param seed: random seed
    :return: subset of dataset
    """
    set_seeds(seed)
