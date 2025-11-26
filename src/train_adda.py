import torch
from torch import nn, optim
import warnings

import torch.nn.functional as F
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from model.adda import ADDA  # 假设你的 ADDA 类保存在 model/adda.py 中
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


def validate_model(classifier, encoder, valid_loader, device):
    """
    🎯 验证模型在目标域上的性能（使用 encoder + classifier）
    """
    classifier.eval()
    encoder.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, _ in valid_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            features = encoder(data)
            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    classifier.train()
    encoder.train()

    print(f"target domain acc: {accuracy:.2f}")
    return accuracy


def train_adda(
    source_loader,
    target_loader,
    num_epochs_stage1=20,
    num_epochs_stage2=50,
    lr_stage1=1e-3,
    lr_stage2=1e-3,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # ========================
    # 第一阶段：源域预训练
    # ========================
    print("=== Stage 1: Pre-training on Source Domain ===")
    model = ADDA(num_classes=11).to(device)

    optimizer_src = optim.Adam(
        list(model.source_encoder.parameters()) + list(model.classifier.parameters()),
        lr=lr_stage1,
        weight_decay=1e-4
    )
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs_stage1):
        src_iter = iter(source_loader)
        for i, (src_data, src_labels, _) in enumerate(src_iter):
            src_data, src_labels = src_data.to(device), src_labels.to(device)

            logits, _ = model.forward_source(src_data)
            loss = ce_loss(logits, src_labels)

            optimizer_src.zero_grad()
            loss.backward()
            optimizer_src.step()

            if i % 20 == 0:
                print(f"Stage1 Epoch [{epoch + 1}/{num_epochs_stage1}] "
                      f"[Batch {i}/{len(source_loader)}] | Cls Loss: {loss.item():.4f}")

    # ========================
    # 第二阶段：对抗自适应
    # ========================
    print("\n=== Stage 2: Adversarial Adaptation ===")

    # 冻结源编码器和分类器
    model.freeze_source_and_classifier()
    # 复制权重到目标编码器
    model.copy_source_to_target()
    # 确保目标编码器和判别器可训练（虽然默认是 True，但显式调用更清晰）
    model.unfreeze_target_and_disc()

    # 初始化优化器（只更新 target_encoder 和 domain_discriminator）
    optimizer_tgt = optim.Adam(model.target_encoder.parameters(), lr=lr_stage2, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(model.domain_discriminator.parameters(), lr=lr_stage2, betas=(0.5, 0.999))
    bce_loss = nn.BCELoss()

    min_len = min(len(source_loader), len(target_loader))

    model.train()
    for epoch in range(num_epochs_stage2):
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        for i in range(min_len):
            # 获取数据
            src_data, _, _ = next(src_iter)
            tgt_data, _, _ = next(tgt_iter)

            min_batch = min(src_data.size(0), tgt_data.size(0))
            src_data = src_data[:min_batch].to(device)
            tgt_data = tgt_data[:min_batch].to(device)

            # ---- 更新域判别器 D ----
            # with torch.no_grad():
            src_feat = model.source_encoder(src_data)   # 固定
            tgt_feat = model.target_encoder(tgt_data)   # 当前

            disc_input = torch.cat([src_feat, tgt_feat], dim=0)
            disc_labels = torch.cat([
                torch.ones(src_feat.size(0), device=device),
                torch.zeros(tgt_feat.size(0), device=device)
            ])
            disc_preds = model.discriminate(disc_input)
            loss_disc = bce_loss(disc_preds, disc_labels)

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # ---- 更新目标编码器 Mt（对抗训练）----
            tgt_feat_adv = model.target_encoder(tgt_data)
            adv_labels = torch.ones(tgt_feat_adv.size(0), device=device)  # 欺骗判别器
            adv_preds = model.discriminate(tgt_feat_adv)
            loss_adv = bce_loss(adv_preds, adv_labels)

            optimizer_tgt.zero_grad()
            loss_adv.backward()
            optimizer_tgt.step()

            if i % 20 == 0:
                print(f"Stage2 Epoch [{epoch + 1}/{num_epochs_stage2}] "
                      f"[Batch {i}/{min_len}] | Disc Loss: {loss_disc.item():.4f} | Adv Loss: {loss_adv.item():.4f}")

        # 每 5 个 epoch 验证一次
        if epoch % 5 == 0:
            validate_model(model.classifier, model.target_encoder, target_loader, device)

    return model.target_encoder, model.classifier


if __name__ == "__main__":
    device = get_device()

    batch_size = 1024
    num_epochs_stage1 = 20
    num_epochs_stage2 = 50

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    train_adda(
        source_train_loader,
        target_train_loader,
        num_epochs_stage1=num_epochs_stage1,
        num_epochs_stage2=num_epochs_stage2,
        device=device
    )