import torch
from torch import nn, optim
import warnings


import torch.nn.functional as F
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from model.dan import DAN
from model import modelutils
# from model.mkmmd_loss import mk_mmd
from model.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from sklearn.metrics import accuracy_score, top_k_accuracy_score

warnings.filterwarnings('ignore')


def train_dan(
        source_loader,
        target_loader,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    model = DAN(num_classes=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()

    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False
    )

    # 获取较短的迭代次数
    min_len = min(len(source_loader), len(target_loader))

    model.train()
    for epoch in range(num_epochs):
        total_cls_loss = 0.0
        total_mmd_loss = 0.0

        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        for i in range(min_len):
            # 获取数据
            src_data, src_labels, _ = next(src_iter)
            tgt_data, _, _ = next(tgt_iter)  # 目标域无标签

            src_data, src_labels = src_data.to(device), src_labels.to(device)
            tgt_data = tgt_data.to(device)

            # 前向传播
            src_logits, src_feats = model(src_data)
            _, tgt_feats = model(tgt_data)

            # 分类损失（仅源域）
            cls_loss = ce_loss(src_logits, src_labels)

            # MMD 损失（多层）
            mmd_loss = 0.0
            for key in src_feats.keys():
                # mmd_loss += mk_mmd(src_feats[key], tgt_feats[key])
                mmd_loss += mkmmd_loss(src_feats[key], tgt_feats[key])

            # 总损失
            total_loss = cls_loss + mmd_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()

        avg_cls = total_cls_loss / min_len
        avg_mmd = total_mmd_loss / min_len
        if i % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] [Batch {i}/{min_len}] | Cls Loss: {avg_cls:.4f} | MMD Loss: {avg_mmd:.4f}")

        if epoch % 5 == 0:
            validate_model(model, target_train_loader, device)

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

    return accuracy

if __name__ == "__main__":

    device: torch.device = get_device()

    batch_size = 64
    num_epochs = 50
    num_k = 4

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    train_dan(source_train_loader, target_train_loader, num_epochs)
