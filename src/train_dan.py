import torch
from torch import nn, optim
import warnings


import torch.nn.functional as F
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from module.dan import ImageClassifier
from module import utils
from module.loss import MultipleKernelMaximumMeanDiscrepancy
from module.kernels import GaussianKernel
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from dataset.dataset_utils import set_seeds

warnings.filterwarnings('ignore')


def train_dan(
        source_loader,
        target_loader,
        da_dataset: str, model_name: str, seq: int,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):

    set_seeds(seq)

    model = ImageClassifier(num_classes=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()

    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False
    )

    # 获取较短的迭代次数
    min_len = min(len(source_loader), len(target_loader))

    model.train()
    best_acc = 0
    for epoch in range(num_epochs):
        total_cls_loss = 0.0
        total_mmd_loss = 0.0

        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        for i in range(min_len):
            # 获取数据
            src_data, src_labels, _ = next(src_iter)
            tgt_data, _, _ = next(tgt_iter)  # 目标域无标签

            min_batch = min(src_data.size(0), tgt_data.size(0))
            src_data, src_labels = src_data[:min_batch], src_labels[:min_batch]
            tgt_data = tgt_data[:min_batch]

            src_data, src_labels = src_data.to(device), src_labels.to(device)
            tgt_data = tgt_data.to(device)

            # 前向传播
            y_s, f_s = model(src_data)
            y_t, f_t = model(tgt_data)

            # 分类损失（仅源域）
            cls_loss = ce_loss(y_s, src_labels)

            # MMD 损失（多层）
            transfer_loss = mkmmd_loss(f_s, f_t)

            # 总损失
            total_loss = cls_loss + transfer_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] [Batch {i}/{min_len}] | Cls Loss: {cls_loss.item():.4f} | MMD Loss: {transfer_loss.item():.4f}")

        acc = validate_model(model, target_loader, device)
        if acc > best_acc:
            best_acc = acc
            print(f"new best acc:{best_acc}, weights saved")
            torch.save(model.state_dict(),
                       f"../autodl-tmp/uda/{da_dataset}/{model_name}/" + f'{model_name}_{seq}.pth')

    return model


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

            class_logits, _ = model(data)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    print(f"target domain acc:{accuracy}")

    return accuracy


if __name__ == "__main__":

    device: torch.device = get_device()

    batch_size = 1024
    num_epochs = 50
    num_k = 4

    loader16a, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    loader22, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    loader16c, _ = DataloaderHelper.dataloader_04c(batch_size, 1.0)

    for i in range(3):
        train_dan(loader16a, loader22, "16a_22", "dan", i, num_epochs)

    for i in range(3):
        train_dan(loader22, loader16a, "22_16a", "dan", i, num_epochs)

    for i in range(3):
        train_dan(loader16c, loader22, "16c_22", "dan", i, num_epochs)

    for i in range(3):
        train_dan(loader22, loader16c, "22_16c", "dan", i, num_epochs)


