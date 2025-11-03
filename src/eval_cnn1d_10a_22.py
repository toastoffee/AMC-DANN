import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import evaluate
from model.cnn1d import CNN1d

warnings.filterwarnings('ignore')


def run_train():

    device: torch.device = get_device()

    batch_size = 512
    valid_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    criterion_ce = nn.CrossEntropyLoss()

    model = CNN1d().to(device)
    model.load_state_dict(torch.load('cnn1d_10a_all.pth'))

    evaluate(model, valid_loader, criterion_ce, device, "[cnn1d_10a_on_22]")


if __name__ == "__main__":
    run_train()
