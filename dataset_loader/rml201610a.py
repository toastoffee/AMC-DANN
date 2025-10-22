import torch
from torch.utils.data import Dataset

import scipy.io as scio
import numpy as np
import utils

dataset_path = "/Users/toastoffee/Documents/Datasets/RML2016.10a_dict.mat"


class RML201610a(Dataset):
    def __init__(self):
        self.class_num = 11
        self.sample_num = 220000

        dataset_dict = scio.loadmat(dataset_path)

        self.X = torch.from_numpy(dataset_dict['X']).float()
        self.X = utils.sgn_norm(self.X)

        self.Y = torch.from_numpy(dataset_dict['Y']).squeeze().t().long()

        self.snr = torch.from_numpy(dataset_dict['snr']).t().float()

        self.modulation = dataset_dict['modulation']

        self.snrs = np.unique(self.snr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.snr[item]


if __name__ == "__main__":
    dataset = RML201610a()