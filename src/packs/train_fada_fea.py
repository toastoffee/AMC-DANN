import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import train_and_evaluate
from model.fada import FADA
from dataset.fada_dataset import FadaDataset
from dataset.rml_dataset import RmlHelper

warnings.filterwarnings('ignore')


def run_train():

    device: torch.device = get_device()

    batch_size = 512
    s_ds = RmlHelper.rml201610a()
    t_ds = RmlHelper.rml22()
    dataset = FadaDataset(s_ds, t_ds, 0.6,  50, 1)
    source_train_dataloader = DataLoader(dataset=dataset.source_train_subset, batch_size=batch_size, shuffle=True)
    source_valid_dataloader = DataLoader(dataset=dataset.source_valid_subset, batch_size=batch_size)

    criterion_ce = nn.CrossEntropyLoss()

    model = FADA().to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    train_and_evaluate(model, source_train_dataloader, source_valid_dataloader, optimizer, criterion_ce, device, 50, "fada_fea_cls")


if __name__ == "__main__":
    run_train()
