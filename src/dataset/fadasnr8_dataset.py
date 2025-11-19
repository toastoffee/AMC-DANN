import random

from torch.utils.data import Dataset, DataLoader
import torch

import dataset_utils
from rml_dataset import RmlHelper


class FadaSnr8Dataset:

    def __init__(self,
                 source_dataset: Dataset,
                 target_dataset: Dataset,
                 train_ratio:    float,
                 shots:          int,
                 seed:           int):
        """
        initialize the dataloader
        :param source_dataset: dataset of source domain
        :param target_dataset: dataset of target domain
        :param train_ratio: train ratio of the datasets
        :param shots: sample count of each class
        :param seed: random seed
        """
        self.shots = shots
        self.split_ratio = [train_ratio, 1.0 - train_ratio]
        self.source_train_subset, self.source_valid_subset = dataset_utils.dataset_random_split(source_dataset, self.split_ratio, seed)
        self.target_train_subset, self.target_valid_subset = dataset_utils.dataset_random_split(target_dataset, self.split_ratio, seed)

        self.target_train_labeled_subset = dataset_utils.get_few_shots_by_snrs_and_class(self.target_train_subset, shots, seed)
        source_train_items = [self.source_train_subset[i] for i in range(len(self.source_train_subset))]
        target_train_labeled_items = [self.target_train_labeled_subset[i] for i in range(len(self.target_train_labeled_subset))]
        self.X_s = torch.Tensor([item[0] for item in source_train_items])
        self.Y_s = torch.Tensor([int(item[1]) for item in source_train_items])
        self.Snr_s = torch.Tensor([item[2] for item in source_train_items])

        self.X_t = torch.Tensor([item[0] for item in target_train_labeled_items])
        self.Y_t = torch.Tensor([int(item[1]) for item in target_train_labeled_items])
        self.Snr_t = torch.Tensor([item[2] for item in target_train_labeled_items])

    def create_pair_groups(self, seed: int):
        """
        get G1,G2,G3,G4 pairs
        G1: a pair of pic comes from same domain , same Snr
        G3: a pair of pic comes from same domain, different snr
        G2: a pair of pic comes from different domain,same snr
        G4: a pair of pic comes from different domain, different snrs
        :param seed: random seed
        :return: group of pairs, and labels
        """
        dataset_utils.set_seeds(seed)

        n = self.X_t.shape[0]   # class_num * snr_num * shots

        # shuffle order
        classes = torch.unique(self.Y_s)
        snrs = torch.unique(self.Snr_t)

        classes = classes[torch.randperm(len(classes))]
        snrs = snrs[torch.randperm(len(snrs))]

        pairs = []
        for class_label in classes:
            for snr in snrs:
                pairs.append((class_label.item(), snr.item()))

        classes_num = classes.shape[0]
        snrs_num = snrs.shape[0]

        def s_rand_indices(pair):
            condition = (self.Y_s == pair[0]) & (self.Snr_s == pair[1])
            idx = torch.nonzero(condition).squeeze()
            return idx[torch.randperm(len(idx))][:self.shots * 2].squeeze()

        def t_indices(pair):
            condition = (self.Y_t == pair[0]) & (self.Snr_t == pair[1])
            idx = torch.nonzero(condition).squeeze()
            if idx.dim() == 0:
                return idx
            else:
                return idx[:self.shots].squeeze()

        source_idxs = list(map(s_rand_indices, pairs))  # [20 * 11, 2*shot]
        target_idxs = list(map(t_indices, pairs))   # [20 * 11, shot]

        source_matrix = torch.stack(source_idxs)   # [20 * 11, 2*shot]
        target_matrix = torch.stack(target_idxs)    # [20 * 11, shot]

        if target_matrix.shape == torch.Size([220]):
            target_matrix = target_matrix.reshape(220, 1)

        # 初始化四组样本对和对应的标签对
        G1, G2, G3, G4, G5, G6, G7, G8 = [], [], [], [], [], [], [], []
        Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8 = [], [], [], [], [], [], [], []

        def diff_class(idx):
            return (idx + snrs_num) % (classes_num * snrs_num)

        def diff_snr(idx):
            return classes_num * (idx // classes_num) + ((idx % classes_num + 1) % classes_num)

        for i in range(snrs_num * classes_num):
            for j in range(self.shots):

                # G1: 同域 同类 同SNR样本对（源域-源域，相同SNR）
                # 从源域的第i个类别中取第j*2和j*2+1个样本组成一对
                G1.append((self.X_s[source_matrix[i][j * 2]], self.X_s[source_matrix[i][j * 2 + 1]]))
                Y1.append((self.Y_s[source_matrix[i][j * 2]], self.Y_s[source_matrix[i][j * 2 + 1]]))

                # G2: 异域 同类 同SNR样本对（源域-目标域，相同SNR）
                # 源域第i类第j个样本 + 目标域第i类第j个样本
                G2.append((self.X_s[source_matrix[i][j]], self.X_t[target_matrix[i][j]]))
                Y2.append((self.Y_s[source_matrix[i][j]], self.Y_t[target_matrix[i][j]]))

                # G3: 同域 异类 同SNR样本对（源域-源域，相同SNR）
                # 源域第i类第j个样本 + 源域第(i+1)%10类第j个样本
                G3.append((self.X_s[source_matrix[i][j]], self.X_s[source_matrix[diff_class(i)][j]]))
                Y3.append((self.Y_s[source_matrix[i][j]], self.Y_s[source_matrix[diff_class(i)][j]]))

                # G4: 异域 异类 同SNR样本对（源域-目标域，相同SNR）
                # 源域第i类第j个样本 + 目标域第(i+1)%10类第j个样本
                G4.append((self.X_s[source_matrix[i][j]], self.X_t[target_matrix[diff_class(i)][j]]))
                Y4.append((self.Y_s[source_matrix[i][j]], self.Y_t[target_matrix[diff_class(i)][j]]))

                # G5: 同域 同类 异SNR样本对（源域-源域，不同SNR）
                # 从源域的第i个类别中取第j*2和j*2+1个样本组成一对
                G5.append((self.X_s[source_matrix[i][j * 2]], self.X_s[source_matrix[diff_snr(i)][j * 2 + 1]]))
                Y5.append((self.Y_s[source_matrix[i][j * 2]], self.Y_s[source_matrix[diff_snr(i)][j * 2 + 1]]))

                # G6: 异域 同类 异SNR样本对（源域-目标域，不同SNR）
                # 源域第i类第j个样本 + 目标域第i类第j个样本
                G6.append((self.X_s[source_matrix[i][j]], self.X_t[target_matrix[diff_snr(i)][j]]))
                Y6.append((self.Y_s[source_matrix[i][j]], self.Y_t[target_matrix[diff_snr(i)][j]]))

                # G7: 同域 异类 异SNR样本对（源域-源域，不同SNR）
                # 源域第i类第j个样本 + 源域第(i+1)%10类第j个样本
                G7.append((self.X_s[source_matrix[i][j]], self.X_s[source_matrix[diff_class(diff_snr(i))][j]]))
                Y7.append((self.Y_s[source_matrix[i][j]], self.Y_s[source_matrix[diff_class(diff_snr(i))][j]]))

                # G8: 异域 异类 异SNR样本对（源域-目标域，不同SNR）
                # 源域第i类第j个样本 + 目标域第(i+1)%10类第j个样本
                G8.append((self.X_s[source_matrix[i][j]], self.X_t[target_matrix[diff_class(diff_snr(i))][j]]))
                Y8.append((self.Y_s[source_matrix[i][j]], self.Y_t[target_matrix[diff_class(diff_snr(i))][j]]))


        # 组合四组样本对和标签对
        groups = [G1, G2, G3, G4, G5, G6, G7, G8]
        groups_y = [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8]

        # 验证每组样本对数量是否一致（都等于n）
        for g in groups:
            assert (len(g) == n)

        return groups, groups_y


if __name__ == "__main__":
    s_ds = RmlHelper.rml201610a()
    t_ds = RmlHelper.rml22()

    loader = FadaSnr8Dataset(s_ds, t_ds, 0.6, 1, 1)
    loader.create_pair_groups(2)