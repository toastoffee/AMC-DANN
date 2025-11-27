import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_dann import train_dann
from model.dann import DANN

warnings.filterwarnings('ignore')


def run_train():
    device: torch.device = get_device()

    batch_size = 512

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)

    model = DANN().to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    train_dann(model, source_train_loader, target_train_loader, target_train_loader, optimizer, device, 100, "dann_10a_22")


if __name__ == "__main__":
    run_train()
