import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio
import torch.nn.functional as F

import datautils
from configuration import Config


rml201610a_path = Config.get("dataset_dir") + Config.get("rml201610a_mat")
rml201604c_path = Config.get("dataset_dir") + Config.get("rml201604c_mat")
rml22_path = Config.get("dataset_dir") + Config.get("rml22_mat")


class RmlDataset(Dataset):
    def __init__(self,
                 dataset_path:     str,
                 dataset_name:     str,
                 has_domain_label: bool = False,
                 domain_label:     int = 0,
                 normalized:       bool = True):

        self.has_domain_label = has_domain_label

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

        if self.has_domain_label:
            self.domain = torch.full((self.sample_num, ), fill_value=domain_label, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.has_domain_label:
            return self.X[item], self.Y[item], self.snr[item], self.domain[item]
        else:
            return self.X[item], self.Y[item], self.snr[item]


class RmlHelper:

    @staticmethod
    def rml201610a(has_domain_label: bool = False,
                   domain_label:     int = 0) -> RmlDataset:
        return RmlDataset(rml201610a_path, "rml201610a",
                          has_domain_label, domain_label)

    @staticmethod
    def rml201604c(has_domain_label: bool = False,
                   domain_label:     int = 0) -> RmlDataset:
        return RmlDataset(rml201604c_path, "rml201604c",
                          has_domain_label, domain_label)

    @staticmethod
    def rml22(has_domain_label: bool = False,
              domain_label:     int = 0) -> RmlDataset:
        return RmlDataset(rml22_path, "rml22",
                          has_domain_label, domain_label)


if __name__ == "__main__":
    # rml2016a = RmlHelper.rml201610a()
    # rml2016c = RmlHelper.rml201604c()
    rml22 = RmlHelper.rml22()
