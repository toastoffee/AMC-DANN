import torch
from torch import nn, optim
import warnings
import torch.nn.functional as F
from train.device_utils import get_device
from model.adda import ADDA
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


def train_adda(
        source_loader,
        target_loader,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    ADDA训练函数 - 遵循DAN代码风格
    """
    # 初始化模型
    model = ADDA(num_classes=11).to(device)

    # 阶段1：源域预训练优化器
    source_optimizer = optim.Adam(
        list(model.source_encoder.parameters()) + list(model.classifier.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # 阶段2：对抗训练优化器
    target_optimizer = optim.Adam(model.target_encoder.parameters(), lr=lr / 10, weight_decay=1e-4)
    disc_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr / 10, weight_decay=1e-4)

    # 损失函数
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print("=== 阶段1: 源域编码器预训练 ===")

    # 阶段1：源域预训练
    model.train()
    for epoch in range(num_epochs // 2):  # 一半epoch用于源预训练
        total_cls_loss = 0.0
        batch_count = 0

        for i, (src_data, src_labels, _) in enumerate(source_loader):
            src_data, src_labels = src_data.to(device), src_labels.to(device)

            # 源域前向传播
            logits_src = model(src_data, domain='source')
            cls_loss = ce_loss(logits_src, src_labels)

            # 反向传播
            source_optimizer.zero_grad()
            cls_loss.backward()
            source_optimizer.step()

            total_cls_loss += cls_loss.item()
            batch_count += 1

            if i % 20 == 0:
                print(f"预训练 Epoch [{epoch + 1}/{num_epochs // 2}] [Batch {i}/{len(source_loader)}] | "
                      f"Cls Loss: {cls_loss.item():.4f}")

        # 每5个epoch验证一次
        if epoch % 5 == 0:
            source_acc = validate_model(model, source_loader, device, domain='source')
            print(f"预训练 Epoch [{epoch + 1}/{num_epochs // 2}] | "
                  f"平均Cls Loss: {total_cls_loss / batch_count:.4f} | "
                  f"源域准确率: {source_acc:.2f}%")

    print("=== 阶段2: 对抗训练 ===")

    # 解冻目标编码器
    for param in model.target_encoder.parameters():
        param.requires_grad = True

    # 获取较短的迭代次数
    min_len = min(len(source_loader), len(target_loader))

    # 阶段2：对抗训练
    for epoch in range(num_epochs // 2, num_epochs):
        total_disc_loss = 0.0
        total_target_loss = 0.0
        batch_count = 0

        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        for i in range(min_len):
            # 获取数据
            src_data, src_labels, _ = next(src_iter)
            tgt_data, _, _ = next(tgt_iter)

            min_batch = min(src_data.size(0), tgt_data.size(0))
            src_data = src_data[:min_batch].to(device)
            tgt_data = tgt_data[:min_batch].to(device)

            # ===== 训练判别器 =====
            model.target_encoder.eval()
            model.discriminator.train()

            # 源域特征（固定）
            with torch.no_grad():
                _, feat_src = model(src_data, domain='source', return_features=True)

            # 目标域特征
            _, feat_tgt = model(tgt_data, domain='target', return_features=True)

            # 判别器预测
            pred_src = model.get_domain_prediction(feat_src.detach())
            pred_tgt = model.get_domain_prediction(feat_tgt.detach())

            # 判别器损失
            loss_disc_src = bce_loss(pred_src, torch.ones_like(pred_src))
            loss_disc_tgt = bce_loss(pred_tgt, torch.zeros_like(pred_tgt))
            loss_disc = 1.0 * (loss_disc_src + loss_disc_tgt) / 2

            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()

            # ===== 训练目标编码器 =====
            model.target_encoder.train()
            model.discriminator.eval()

            # 目标域特征
            _, feat_tgt = model(tgt_data, domain='target', return_features=True)
            pred_tgt = model.get_domain_prediction(feat_tgt)

            # 目标编码器损失（倒置标签）
            loss_target = 1.0 * bce_loss(pred_tgt, torch.ones_like(pred_tgt))

            target_optimizer.zero_grad()
            loss_target.backward()
            target_optimizer.step()

            total_disc_loss += loss_disc.item()
            total_target_loss += loss_target.item()
            batch_count += 1

            if i % 20 == 0:
                print(f"对抗训练 Epoch [{epoch + 1}/{num_epochs}] [Batch {i}/{min_len}] | "
                      f"Disc Loss: {loss_disc.item():.4f} | Target Loss: {loss_target.item():.4f}")

        # 每5个epoch验证一次
        if epoch % 5 == 0:
            target_acc = validate_model(model, target_loader, device, domain='target')
            print(f"对抗训练 Epoch [{epoch + 1}/{num_epochs}] | "
                  f"平均Disc Loss: {total_disc_loss / batch_count:.4f} | "
                  f"平均Target Loss: {total_target_loss / batch_count:.4f} | "
                  f"目标域准确率: {target_acc:.2f}%")

    return model


def validate_model(model, valid_loader, device, domain: str = 'target'):
    """
    🎯 验证模型性能 - 适配ADDA的双编码器结构
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, snr in valid_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # 根据域选择编码器
            logits = model(data, domain=domain)
            _, predicted = torch.max(logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy


if __name__ == "__main__":
    from dataset.dataloader_helper import DataloaderHelper

    device: torch.device = get_device()

    batch_size = 1024
    num_epochs = 50

    # 加载数据
    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    # 训练ADDA模型
    trained_model = train_adda(
        source_train_loader,
        target_train_loader,
        num_epochs=num_epochs,
        device=device
    )

    # 最终验证
    final_acc = validate_model(trained_model, target_train_loader, device, domain='target')
    print(f"🎯 ADDA训练完成！最终目标域准确率: {final_acc:.2f}%")