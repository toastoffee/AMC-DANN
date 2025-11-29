import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
from dataset.dataset_utils import set_seeds
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_dann import train_dann
from module.dann import DANN

warnings.filterwarnings('ignore')


def run_train(source_train_loader, target_train_loader, da_dataset: str, model_name: str, seq: int):
    set_seeds(seq)

    device: torch.device = get_device()

    model = DANN().to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    train_dann(model, source_train_loader, target_train_loader, target_train_loader, optimizer, device, 20, da_dataset, model_name, seq)


if __name__ == "__main__":
    batch_size = 256

    # source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0, True, 0)
    # target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)
    # for i in range(3):
    #     run_train(source_train_loader, target_train_loader, "16a_22", "sdidn", i)

    # 22->16a
    source_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0, True, 1)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "22_16a", "dann", i)

    # 16c->22
    source_train_loader, _ = DataloaderHelper.dataloader_04c(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "16c_22", "dann", i)

    # 22->16c
    source_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_04c(batch_size, 1.0, True, 1)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "22_16c", "dann", i)
