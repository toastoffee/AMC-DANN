from torch.utils.data import Dataset, DataLoader
import torch

from dataloader_helper import DataloaderHelper
from rml_dataset import RmlHelper


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
#
#     @staticmethod
#     def get_labeled_samples(dataset: Dataset,
#                             shot:    int):
#         from collections import defaultdict
#         import random
#         from torch.utils.data import Subset
#
#         class_indices = defaultdict(list)
#         for idx in range(len(dataset)):
#             _, label, _ = dataset[idx]
#             label = int(label)
#             class_indices[label].append(idx)
#
#         selected_indices = []
#         for cls, indices in class_indices.items():
#             # 确保每个类别都采样shot个样本
#             cls_selected = random.sample(indices, shot)
#             selected_indices.extend(cls_selected)
#
#         assert len(selected_indices) == len(class_indices) * shot
#
#         return Subset(dataset, selected_indices)

class FadaDataloader:

    def __init__(self,
                 source_dataset: Dataset,
                 target_dataset: Dataset,
                 train_ratio:    float,
                 ):



if __name__ == "__main__":
    FadaDataloaderHelper.dataloader_fada(1, 0.6)