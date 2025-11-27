import torch
from torch import nn, optim
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import eval_and_get_acc

from module.dann import DANN, DANN_wrapper
from module.dan import DAN, DAN_wrapper
from module.adda import ADDA
from module.mcc import MCC
from module.mcd import MCD, MCD_wrapper
from module.sdidn import DistanDANN, SDIDN_wrapper

from module.loss import CovarianceOrthogonalLoss, DomainContrastiveLoss
from module.grl import GradientReversalFunction
from module.utils import freeze, unfreeze
from dataset.dataset_utils import set_seeds


def draw_fig(da_dataset: str):
    set_seeds(0)

    device = torch.device('cuda:0')

    batch_size = 1024
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    loss_fn = nn.CrossEntropyLoss()

    # 1.sdidn
    sdidn = DistanDANN().to(device)
    sdidn.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/sdidn/" + f'sdidn_1.pth'))
    sdidn = SDIDN_wrapper(sdidn).to(device)
    sdidn.eval()
    sdidn_acc = eval_and_get_acc(sdidn, target_train_loader, loss_fn, device)

    # 2.adda
    adda = ADDA().to(device)
    adda.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/adda/" + f'adda_1.pth'))
    adda.eval()
    adda_acc = eval_and_get_acc(adda, target_train_loader, loss_fn, device)

    # 3.dann
    dann = DANN().to(device)
    dann.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/dann/" + f'dann_1.pth'))
    dann = DANN_wrapper(dann).to(device)
    dann.eval()
    dann_acc = eval_and_get_acc(dann, target_train_loader, loss_fn, device)

    # 4.dan
    dan = DAN().to(device)
    dan.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/dan/" + f'dan_1.pth'))
    dan = DAN_wrapper(dan).to(device)
    dan.eval()
    dan_acc = eval_and_get_acc(dan, target_train_loader, loss_fn, device)

    # 5.mcc
    mcc = MCC().to(device)
    mcc.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/mcc/" + f'mcc_1.pth'))
    mcc.eval()
    mcc_acc = eval_and_get_acc(mcc, target_train_loader, loss_fn, device)

    # 6.mcd
    mcd = MCD().to(device)
    mcd.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/mcd/" + f'mcd_1.pth'))
    mcd = MCD_wrapper(mcd).to(device)
    mcd.eval()
    mcd_acc = eval_and_get_acc(mcd, target_train_loader, loss_fn, device)



if __name__ == "__main__":
