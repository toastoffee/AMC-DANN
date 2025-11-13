from torch.utils.data import Dataset, DataLoader
import torch

import dataset_utils


# class FadaDataloaderHelper:
#     @staticmethod
#     def dataloader_fada(shot:           int,
#                         train_ratio:    float = 0.6):
#
#         source_dataset = RmlHelper.rml201610a()
#         target_dataset = RmlHelper.rml22()
#
#         source_train_dataset, source_valid_dataset = DataloaderHelper.dataset_random_split(source_dataset, train_ratio)
#         target_train_dataset, target_valid_dataset = DataloaderHelper.dataset_random_split(target_dataset, train_ratio)
#
#         target_labeled_subset = FadaDataloaderHelper.get_labeled_samples(target_train_dataset, 1)
#
#

class FadaDataloader:

    def __init__(self,
                 source_dataset: Dataset,
                 target_dataset: Dataset,
                 train_ratio:    float,
                 batch_size:     int,
                 shots:          int,
                 seed:           int):
        """
        initialize the dataloader
        :param source_dataset: dataset of source domain
        :param target_dataset: dataset of target domain
        :param train_ratio: train ratio of the datasets
        :param batch_size: batch size
        :param shots: sample count of each class
        :param seed: random seed
        """
        self.shots = shots
        self.split_ratio = [train_ratio, 1.0 - train_ratio]
        self.batch_size = batch_size
        self.source_train_subset, self.source_valid_subset = dataset_utils.dataset_random_split(source_dataset, self.split_ratio, seed)
        self.target_train_subset, self.target_valid_subset = dataset_utils.dataset_random_split(target_dataset, self.split_ratio, seed)
        self.target_labeled_subset = dataset_utils.get_few_shots(self.target_train_subset, self.shots, seed)

    def create_pair_groups(self, seed: int):
        """
        get G1,G2,G3,G4 pairs
        G1: a pair of pic comes from same domain ,same class
        G3: a pair of pic comes from same domain, different classes
        G2: a pair of pic comes from different domain,same class
        G4: a pair of pic comes from different domain, different classes
        :param seed: random seed
        :return: group of pairs, and labels
        """
        source_labeled_subset = dataset_utils.get_few_shots(self.source_train_subset, self.shots * 2, seed)




if __name__ == "__main__":
    # FadaDataloaderHelper.dataloader_fada(1, 0.6)