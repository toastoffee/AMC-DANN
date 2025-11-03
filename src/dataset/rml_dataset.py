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
rml22_path = Config.get("dataset_dir") + Config.get("rml22_mat")

class RmlDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_name: str,
                 normalized: bool = True):

        dataset_dict = scio.loadmat(dataset_path)

        self.dataset_name = dataset_name

        self.X = torch.from_numpy(dataset_dict['X']).float()

        if normalized:
            self.X = datautils.sgn_norm(self.X)

        self.Y = torch.from_numpy(dataset_dict['Y']).squeeze().t().long()

        self.snr = torch.from_numpy(dataset_dict['snr']).t().float()

        self.modulation = dataset_dict['modulation']

        self.class_num = len(np.unique(self.modulation))
        self.sample_num = self.X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item], self.snr[item]


class RmlHelper:

    @staticmethod
    def rml201610a() -> RmlDataset:
        return RmlDataset(rml201610a_path, "rml201610a")

    @staticmethod
    def rml201604c() -> RmlDataset:
        return RmlDataset(rml201604c_path, "rml201604c")

    @staticmethod
    def rml22() -> RmlDataset:
        return RmlDataset(rml22_path, "rml22")


if __name__ == "__main__":
    # rml2016a = RmlHelper.rml201610a()
    # rml2016c = RmlHelper.rml201604c()
    rml22 = RmlHelper.rml22()
