import torch
from torch import nn, optim
import warnings

import matplotlib.pyplot as plt
import itertools
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import eval_and_get_acc

from module.dann import DANN, DANN_wrapper
from module.dan import ImageClassifier, DAN_wrapper
from module.adda import ADDA
from module.mcc import MCC
from module.mcd import MCD, MCD_wrapper
from module.sdidn import DistanDANN, SDIDN_wrapper

from module.loss import CovarianceOrthogonalLoss, DomainContrastiveLoss
from module.grl import GradientReversalFunction
from module.utils import freeze, unfreeze
from dataset.dataset_utils import set_seeds
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os


def save_accuracies(target_train_loader, seq_count, da_dataset: str, save_file: str = "./results/accuracies.pkl"):
    """
    评估多个 UDA 模型，并将 SNR-wise 准确率保存到文件。
    """
    set_seeds(0)
    device = torch.device('cuda:0')
    loss_fn = nn.CrossEntropyLoss()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    results = {}

    # 1. SDIDN
    sdidn = DistanDANN().to(device)
    results['SDIDN'] = []
    for i in range(seq_count):
        sdidn.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/sdidn/sdidn_{i}.pth"))
        sdidn = SDIDN_wrapper(sdidn).to(device)
        sdidn.eval()
        results['SDIDN'].append(eval_and_get_acc(sdidn, target_train_loader, loss_fn, device))

    # 2. ADDA
    adda = ADDA().to(device)
    results['ADDA'] = []
    for i in range(seq_count):
        adda.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/adda/adda_1.pth"))
        adda.eval()
        results['ADDA'].append(eval_and_get_acc(adda, target_train_loader, loss_fn, device))

    # 3. DANN
    dann = DANN().to(device)
    results['DANN'] = []
    for i in range(seq_count):
        dann.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/dann/dann_1.pth"))
        dann = DANN_wrapper(dann).to(device)
        dann.eval()
        results['DANN'].append(eval_and_get_acc(dann, target_train_loader, loss_fn, device))

    # 4. DAN
    dan = ImageClassifier().to(device)
    results['DAN'] = []
    for i in range(seq_count):
        dan.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/dan/dan_1.pth"))
        dan = DAN_wrapper(dan).to(device)
        dan.eval()
        results['DAN'].append(eval_and_get_acc(dan, target_train_loader, loss_fn, device))

    # 5. MCC
    mcc = MCC(num_classes=11).to(device)
    results['MCC'] = []
    for i in range(seq_count):
        mcc.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/mcc/mcc_1.pth"))
        mcc.eval()
        results['MCC'].append(eval_and_get_acc(mcc, target_train_loader, loss_fn, device))

    # 6. MCD
    mcd = MCD().to(device)
    results['MCD'] = []
    for i in range(seq_count):
        mcd.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/mcd/mcd_1.pth"))
        mcd = MCD_wrapper(mcd).to(device)
        mcd.eval()
        results['MCD'].append(eval_and_get_acc(mcd, target_train_loader, loss_fn, device))

    # 保存为 pickle 文件（支持 dict、tensor 等）
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Accuracies saved to {save_file}")


if __name__ == "__main__":
    batch_size = 1024
    loader22, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    save_accuracies(loader22, 5, "16a_22", "./results/16a_22_accs.pkl")
