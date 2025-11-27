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

    shots_selections = [1, 2, 5, 10, 20, 50, 100, 200]
    rounds = 3
    for shots in shots_selections:
        avg_acc = 0
        avg_acc5 = 0
        for i in range(rounds):
            model.load_state_dict(torch.load(f'fada_weights/fadasnr_shots-{shots}_round-{i}.pth'))
            acc, loss, acc5 = evaluate(model, target_valid_dataloader, criterion_ce, device, f"[fada_shots-{shots}_round-{i}]")
            avg_acc += acc()
            avg_acc5 += acc5()

        avg_acc /= float(rounds)
        avg_acc5 /= float(rounds)
        print(f"acc:{avg_acc}, acc5:{avg_acc5}")


if __name__ == "__main__":
    run_train()
