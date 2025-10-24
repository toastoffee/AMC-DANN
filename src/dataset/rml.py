import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio

# import os
# import sys
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

import datautils
from configuration import Config


rml201610a_path = Config.get("dataset_dir") + Config.get("rml201610a_mat")
rml201604c_path = Config.get("dataset_dir") + Config.get("rml201604c_mat")


class RmlDataset(Dataset):
    def __init__(self, dataset_path: str):

        dataset_dict = scio.loadmat(dataset_path)

        self.X = torch.from_numpy(dataset_dict['X']).float()
        self.X = datautils.sgn_norm(self.X)

        self.Y = torch.from_numpy(dataset_dict['Y']).squeeze().t().long()

        self.snr = torch.from_numpy(dataset_dict['snr']).t().float()

        self.modulation = dataset_dict['modulation']

        self.class_num = len(np.unique(self.modulation))
        self.sample_num = 220000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.snr[item]


class DatasetHelper:
    @staticmethod
    def get_rml201610a() -> RmlDataset:
        return RmlDataset(rml201610a_path)

    @staticmethod
    def get_rml201604c() -> RmlDataset:
        return RmlDataset(rml201604c_path)


if __name__ == "__main__":
    dataset = RmlDataset(rml201610a_path)