import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def set_seeds(seed: int):
    """
    set seed of pytorch and numpy
    :param seed: the random seed
    :return: void
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
                  n:       int,
                  seed:    int) -> Subset:
    """
    randomly pick samples from dataset and return a subset
    :param dataset: dataset to select from
    :param n: the count of samples selected
    :param seed: random seed
    :return: subset of dataset
    """
    if n > len(dataset):
        raise ValueError(f"n is larger than dataset size")

    set_seeds(seed)

    indices = random.sample(range(len(dataset)), n)

    return Subset(dataset, indices)


def shuffle_dataset(dataset: Dataset,
                    seed:    int) -> Subset:
    """
    return the shuffled subset of dataset
    :param dataset: dataset to shuffle
    :param seed: random seed
    :return: subset of the shuffled dataset
    """
    set_seeds(seed)

    indices = np.random.permutation(len(dataset))
    return Subset(dataset, indices)


def get_few_shots(dataset: Dataset,
                  shot:    int,
                  seed:    int) -> Subset:
    """
    get random samples, shot for each class
    :param dataset: dataset pick from
    :param shot: sample count of each class
    :param seed: random seed
    :return: subset of few shots
    """
    set_seeds(seed)

    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        label = int(label)
        class_indices[label].append(idx)

    selected_indices = []
    for cls, indices in class_indices.items():
        # 确保每个类别都采样shot个样本
        cls_selected = random.sample(indices, shot)
        selected_indices.extend(cls_selected)

    assert len(selected_indices) == len(class_indices) * shot

    return Subset(dataset, selected_indices)


def get_few_shots_by_snrs(dataset: Dataset,
                          shot:    int,
                          seed:    int) -> Subset:
    """
    get random samples, shot for each snr
    :param dataset: dataset pick from
    :param shot: sample count of each class
    :param seed: random seed
    :return: subset of few shots
    """
    set_seeds(seed)

    snr_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, snr = dataset[idx]
        snr = int(snr)

        if snr != 20:
            snr_indices[snr].append(idx)

    selected_indices = []
    for cls, indices in snr_indices.items():
        # 确保每个snr都采样shot个样本
        cls_selected = random.sample(indices, shot)
        selected_indices.extend(cls_selected)

    assert len(selected_indices) == len(snr_indices) * shot

    return Subset(dataset, selected_indices)