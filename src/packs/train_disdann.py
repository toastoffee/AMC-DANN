import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os
import tqdm

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_dann import train_dann
from model.dis_dann import DisDANN
from model.info_nce import InfoNCE, domain_aware_contrastive_loss
from model.loss_utils import covariance_orthogonal_loss

warnings.filterwarnings('ignore')


def orthogonal_between_two_models(model1, model2):
    """
    Compute orthogonality loss between corresponding weight matrices of two models.
    Encourages <W1_i, W2_i> ≈ 0 for each layer i.

    Assumes model1 and model2 have same architecture and parameter order.
    """
    loss = 0.0
    count = 0

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        # Only apply to weight parameters (skip bias, BN params, etc.)
        if 'weight' in name1 and param1.ndim >= 2:  # usually weight tensors are 2D+
            # Flatten both weights to vectors
            w1 = param1.view(-1)
            w2 = param2.view(-1)

            # Inner product (should be close to 0)
            inner_prod = torch.dot(w1, w2)
            loss += inner_prod ** 2  # squared to make it smooth and non-negative
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=next(model1.parameters()).device)

    return loss / count  # optional: normalize by number of layers


def run_train():
    device: torch.device = get_device()

    batch_size = 512
    num_epochs = 50

    source_train_loader, source_valid_loader = DataloaderHelper.dataloader_10a(batch_size, 0.6, True, 0)
    target_train_loader, target_valid_loader = DataloaderHelper.dataloader_22(batch_size, 0.6, True, 1)

    model = DisDANN().to(device)

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    model.to(device)
    model.train()
    orthogonal_loss_fn = covariance_orthogonal_loss
    contrastive_loss_fn = InfoNCE(temperature=0.1).to(device)
    classification_loss_fn = nn.CrossEntropyLoss().to(device)

    # create combined dataloader
    min_len = min(len(source_train_loader), len(target_train_loader))
    combined_loader = zip(
        iter(source_train_loader),
        iter(target_train_loader))

    # 训练记录
    train_losses = {
        'total': [], 'class': [], 'orthogonal': [], 'contrastive': []
    }

    # step1: train the class feature extractor and classifier


    # step2: train the entire network
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'class': 0, 'orthogonal': 0, 'contrastive': 0}

        combined_loader = zip(
            iter(source_train_loader),
            iter(target_train_loader))
        for batch_idx, ((source_data, source_labels, source_snr, source_domains),
                        (target_data, target_labels, target_snr, target_domains)) in enumerate(combined_loader):

            if batch_idx >= min_len:
                break

            source_data = source_data.to(device, dtype=torch.float32)
            source_labels = source_labels.to(device)
            source_domains = source_domains.to(device)

            target_data = target_data.to(device, dtype=torch.float32)
            target_labels = target_labels.to(device)
            target_domains = target_domains.to(device)

            batch_size = source_data.size(0)
            combined_data = torch.cat([source_data, target_data], dim=0)
            combined_domains = torch.cat([source_domains, target_domains], dim=0)

            # forward
            class_logits, domain_logits, feature_domain, feature_class = model(combined_data)

            # calculate modulation classification loss (only source-domain)
            source_class_logits = class_logits[:batch_size]
            class_loss = classification_loss_fn(source_class_logits, source_labels)
            domain_loss = classification_loss_fn(domain_logits, combined_domains)

            # 特征正交损失
            orthogonal_loss = orthogonal_loss_fn(feature_domain, feature_class)
            # orthogonal_loss = orthogonal_between_two_models(model.fe_class, model.fe_domain)

            # 对比学习损失
            contrastive_loss = domain_aware_contrastive_loss(feature_domain, combined_domains, contrastive_loss_fn, batch_size)

            lambda_orth = 0.1
            lambda_cont = 0.5
            # total_loss = (
            #     class_loss +
            #     domain_loss +
            #     lambda_orth * orthogonal_loss +
            #     lambda_cont * contrastive_loss
            # )

            total_loss = class_loss + domain_loss

            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['class'] += class_loss.item()
            # epoch_losses['orthogonal'] += orthogonal_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()

            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{min_len}], '
                      f'Total Loss: {total_loss.item():.4f}, Class Loss: {class_loss.item():.4f}, '
                      f'domain class Loss: {domain_loss.item():.4f}, '
                      f'Orthogonal Loss: {orthogonal_loss.item():.4f}, '
                      f'Contrastive Loss: {contrastive_loss.item():.4f}')

        # 🎯 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= min_len
            train_losses[key].append(epoch_losses[key])

        if (epoch + 1) % 5 == 0:
            valid_accuracy = validate_model(model, target_valid_loader, device)
            print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {valid_accuracy:.2f}%')


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

            class_logits, _, _, _ = model(data)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy


if __name__ == "__main__":
    run_train()
