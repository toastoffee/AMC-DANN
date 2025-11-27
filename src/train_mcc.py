import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os
import tqdm

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from module.mcc import MCC
from dataset.dataset_utils import set_seeds
from module.loss import MinimumClassConfusionLoss

warnings.filterwarnings('ignore')


def run_train(da_dataset: str, model_name: str, seq: int):
    device: torch.device = get_device()

    set_seeds(seq)

    batch_size = 512
    num_epochs = 50

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    model = MCC(num_classes=11).to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    model.to(device)
    model.train()
    classification_loss_fn = nn.CrossEntropyLoss().to(device)
    mcc_fn = MinimumClassConfusionLoss(2.5).to(device)

    min_len = min(len(source_train_loader), len(target_train_loader))

    best_acc = 0
    for epoch in range(num_epochs):
        combined_loader = zip(
            iter(source_train_loader),
            iter(target_train_loader))
        for batch_idx, ((source_data, source_labels, source_snr),
                        (target_data, target_labels, target_snr)) in enumerate(combined_loader):

            if batch_idx >= min_len:
                break

            data_s = source_data.to(device, dtype=torch.float32)
            label_s = source_labels.to(device)
            data_t = target_data.to(device, dtype=torch.float32)
            data = torch.cat([data_s, data_t], dim=0)

            y_s = model(data_s)
            y_t = model(data_t)

            class_loss = classification_loss_fn(y_s, label_s)
            transfer_loss = mcc_fn(y_t)
            loss = class_loss + transfer_loss * 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{min_len}], '
                      f'Total Loss: {loss.item():.4f}, Class Loss: {class_loss.item():.4f}, '
                      f'transfer Loss: {transfer_loss.item():.4f}, ')

        valid_accuracy = validate_model(model, target_train_loader, device)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {valid_accuracy:.2f}%')
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            print(f"new best acc:{best_acc}, weights saved")
            torch.save(model.state_dict(), f"../autodl-tmp/uda/{da_dataset}/{model_name}/" + f'{model_name}_{seq}.pth')


def validate_model(model, valid_loader, device):
    """
    🎯 验证模型在源域上的性能
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, snr in valid_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            class_logits = model(data)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy


if __name__ == "__main__":
    for i in range(5):
        run_train("16a_22", "mcc", i)
