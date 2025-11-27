import torch
from torch import nn, optim
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from module.dann import DANN
from module.loss import CovarianceOrthogonalLoss, DomainContrastiveLoss
from module.grl import GradientReversalFunction
from module.utils import freeze, unfreeze
from dataset.dataset_utils import set_seeds

warnings.filterwarnings('ignore')


def run_eval(target_train_loader, da_dataset: str, model_name: str, seq: int):
    device: torch.device = get_device()

    set_seeds(seq)

    model = DANN().to(device)

    # load pretrained weights
    model.load_state_dict(torch.load(f"../autodl-tmp/uda/{da_dataset}/{model_name}/" + f'{model_name}_{seq}.pth'))
    model.eval()

    acc = validate_model(model, target_train_loader, device)

    print(f"eval {da_dataset}|{model_name}|{seq}: acc:{acc}")


def validate_model(model, valid_loader, device):
    """
    🎯 验证模型在源域上的性能
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, snr, domains in valid_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            class_logits, _ = model(data, 1.0)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy


if __name__ == "__main__":
    batch_size = 1024
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)

    for i in range(5):
        run_eval(target_train_loader, "16a_22", "dann", i)