import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import evaluate
from model.dann import DANN, dann_wrapper

warnings.filterwarnings('ignore')


def run_train():
    device: torch.device = get_device()

    batch_size = 512

    target_train_loader, target_valid_loader = DataloaderHelper.dataloader_22(batch_size, 0.6, False, 1)

    criterion_ce = nn.CrossEntropyLoss()

    dann = DANN().to(device)
    dann.load_state_dict(torch.load('dann_10a_22.pth'))
    model = dann_wrapper(dann)

    evaluate(model, target_valid_loader, criterion_ce, device, "[dann_10a_on_22]")


if __name__ == "__main__":
    run_train()
