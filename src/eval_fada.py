import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import evaluate

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
    target_valid_dataloader = DataLoader(dataset=dataset.target_valid_subset, batch_size=batch_size)

    criterion_ce = nn.CrossEntropyLoss()

    model = FADA().to(device)
    model.load_state_dict(torch.load('cnn1d_04c_all.pth'))

    evaluate(model, target_valid_dataloader, criterion_ce, device, "[cnn1d_04c_on_22]")


if __name__ == "__main__":
    run_train()
