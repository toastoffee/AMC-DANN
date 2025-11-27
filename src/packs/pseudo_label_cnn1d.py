import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import evaluate
from model.cnn1d import CNN1d
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def run_train():

    device: torch.device = get_device()

    batch_size = 512
    valid_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)


    model = CNN1d().to(device)
    model.load_state_dict(torch.load('cnn1d_10a_all.pth'))
    model.eval()  # 切换到评估模式

    high_confidence_count = 0
    correct_high_confidence = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels, _ in valid_loader:
            data = data.to(device)
            labels = labels.to(device)  # shape: [B]

            outputs = model(data)  # logits, shape: [B, num_classes]
            probs = F.softmax(outputs, dim=1)  # [B, num_classes]
            max_probs, preds = torch.max(probs, dim=1)  # [B], [B]

            # 找出置信度 > 0.9 的样本
            high_conf_mask = max_probs > 0.9  # boolean tensor [B]

            num_high = high_conf_mask.sum().item()
            high_confidence_count += num_high
            total_samples += data.size(0)

            if num_high > 0:
                # 在高置信样本中，统计预测正确的数量
                correct = (preds == labels)[high_conf_mask].sum().item()
                correct_high_confidence += correct

    # 计算结果
    if high_confidence_count > 0:
        pseudo_label_acc = correct_high_confidence / high_confidence_count
    else:
        pseudo_label_acc = 0.0

    print(f"Total samples in valid set: {total_samples}")
    print(f"High-confidence samples (prob > 0.9): {high_confidence_count}")
    print(f"Correct among them: {correct_high_confidence}")
    print(f"Pseudo-label accuracy: {pseudo_label_acc:.4f} ({pseudo_label_acc * 100:.2f}%)")


if __name__ == "__main__":
    run_train()
