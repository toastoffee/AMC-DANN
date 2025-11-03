import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import train_and_evaluate
from model.cnn1d import CNN1d

warnings.filterwarnings('ignore')


def run_train():

    device: torch.device = get_device()

    batch_size = 512
    train_loader, valid_loader = DataloaderHelper.dataloader_04c(batch_size)

    criterion_ce = nn.CrossEntropyLoss()

    model = CNN1d().to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion_ce, device, 50, "cnn1d_04c_all")


if __name__ == "__main__":
    run_train()
