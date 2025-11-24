import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os
import tqdm

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_dann import train_dann
from model.dual_dann import DualDANN
from model.info_nce import InfoNCE, domain_aware_contrastive_loss
from model.loss_utils import covariance_orthogonal_loss
from model import modelutils
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def build_domain_positive_pairs(feature_domain):
    """
    Build positive pairs for domain contrastive learning.

    Assumes:
        feature_domain: [2N, D]
        first N samples: source domain
        last N samples: target domain

    Returns:
        query: [2N, D]
        positive_key: [2N, D]  # each sample's positive is another from same domain
    """
    N = feature_domain.size(0) // 2

    # Split
    src_feat = feature_domain[:N]  # [N, D]
    tgt_feat = feature_domain[N:]  # [N, D]

    # Circular shift: x[i] -> x[(i+1) % N]
    src_pos = torch.cat([src_feat[1:], src_feat[:1]], dim=0)  # [N, D]
    tgt_pos = torch.cat([tgt_feat[1:], tgt_feat[:1]], dim=0)  # [N, D]

    # Combine
    query = feature_domain  # [2N, D]
    positive_key = torch.cat([src_pos, tgt_pos], dim=0)  # [2N, D]

    return query, positive_key

def build_domain_negative_pairs(feature_domain):
    """
    Build positive pairs for domain contrastive learning.

    Assumes:
        feature_domain: [2N, D]
        first N samples: source domain
        last N samples: target domain

    Returns:
        query: [2N, D]
        positive_key: [2N, D]  # each sample's positive is another from same domain
    """
    N = feature_domain.size(0) // 2

    # Split
    src_feat = feature_domain[:N]  # [N, D]
    tgt_feat = feature_domain[N:]  # [N, D]

    # Circular shift: x[i] -> x[(i+1) % N]
    src_pos = torch.cat([src_feat[N-1:], src_feat[:N-1]], dim=0)  # [N, D]
    tgt_pos = torch.cat([tgt_feat[N-1:], tgt_feat[:N-1]], dim=0)  # [N, D]

    # Combine
    query = feature_domain  # [2N, D]
    negative_key = torch.cat([tgt_pos, src_pos], dim=0)  # [2N, D]

    return query, negative_key


def domain_contrastive_loss(features, temperature=0.1, eps=1e-8):
    """
    Domain-aware contrastive loss:
      - Pull samples from the same domain together
      - Push samples from different domains apart

    Args:
        features: [2N, D] — first N: source, last N: target
        temperature: float > 0
        eps: small constant for numerical stability

    Returns:
        loss: scalar tensor
    """
    device = features.device
    N2 = features.size(0)
    assert N2 % 2 == 0, "Batch size must be even (N source + N target)"
    N = N2 // 2

    # L2 normalize features (critical for cosine similarity)
    features = F.normalize(features, p=2, dim=1)  # [2N, D]

    # Compute cosine similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # [2N, 2N]

    # Build masks
    # Same-domain mask (excluding self)
    same_domain_mask = torch.zeros(N2, N2, dtype=torch.bool, device=device)
    same_domain_mask[:N, :N] = True  # source-source
    same_domain_mask[N:, N:] = True  # target-target
    same_domain_mask.fill_diagonal_(False)  # exclude self

    # Cross-domain mask (all source<->target pairs)
    cross_domain_mask = ~same_domain_mask.clone()
    cross_domain_mask.fill_diagonal_(False)  # though already false

    # Numerically stable log-softmax style computation
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()

    exp_sim = torch.exp(sim_stable)

    # Positive sum: same domain
    pos_sum = (exp_sim * same_domain_mask.float()).sum(dim=1)  # [2N]

    # Negative sum: cross domain
    neg_sum = (exp_sim * cross_domain_mask.float()).sum(dim=1)  # [2N]

    # Avoid division by zero
    denominator = pos_sum + neg_sum + eps
    loss_per_sample = -torch.log((pos_sum + eps) / denominator)

    return loss_per_sample.mean()

def run_train():
    device: torch.device = get_device()

    batch_size = 1024

    num_epochs = 50

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)

    model = DualDANN().to(device)

    # load pretrained weights
    # model.load_state_dict(torch.load('cnn1d_04c_all.pth'))

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    model.to(device)
    model.train()

    modelutils.freeze(model.domain_fe)
    modelutils.freeze(model.domain_classifier)

    orthogonal_loss_fn = covariance_orthogonal_loss
    # contrastive_loss_fn = InfoNCE(temperature=0.1).to(device)
    contrastive_criterion = InfoNCE(temperature=0.1, reduction='mean', negative_mode='unpaired')
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
            # orthogonal_loss = orthogonal_loss_fn(feature_domain, feature_class)

            # 对比学习损失
            # contrastive_loss = domain_aware_contrastive_loss(feature_domain, combined_domains, contrastive_loss_fn, batch_size)
            # query, pos = build_domain_positive_pairs(feature_domain)
            # query, neg = build_domain_negative_pairs(feature_domain)
            # contrastive_loss = contrastive_criterion(query, pos, neg)
            contrastive_loss = domain_contrastive_loss(feature_domain)

            lambda_orth = 0.1
            lambda_cont = 0.5
            # total_loss = (
            #     class_loss +
            #     lambda_orth * orthogonal_loss +
            #     lambda_cont * contrastive_loss
            # )

            total_loss = class_loss + domain_loss + lambda_cont * contrastive_loss

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
                print(f'[Step2]Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{min_len}], '
                      f'Total Loss: {total_loss.item():.4f}, Class Loss: {class_loss.item():.4f}, '
                      f'domain class Loss: {domain_loss.item():.4f}, '
                      # f'Orthogonal Loss: {orthogonal_loss.item():.4f}, '
                      f'Contrastive Loss: {contrastive_loss.item():.4f}')

        # 🎯 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= min_len
            train_losses[key].append(epoch_losses[key])

        valid_accuracy = validate_model(model, target_train_loader, device)
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
