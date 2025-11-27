import torch
from torch import nn, optim
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from model.distan_dann import DistanDANN
from model.loss_utils import covariance_orthogonal_loss, domain_contrastive_loss, covariance_orthogonal_loss_3d
from model.modelutils import GradientReversalFunction, freeze, unfreeze
from dataset.dataset_utils import set_seeds

warnings.filterwarnings('ignore')


def run_train(model_name: str, seq: int):
    device: torch.device = get_device()

    set_seeds(seq)

    batch_size = 1024
    num_epochs = 50

    source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0, True, 0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0, True, 1)

    model = DistanDANN().to(device)

    # load pretrained weights
    model.g.load_state_dict(torch.load('pretrained_distan_encoder.pth'))
    model.reconstructor.load_state_dict(torch.load('pretrained_distan_decoder.pth'))

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-3)

    model.to(device)
    model.train()
    freeze(model.g)
    # freeze(model.reconstructor)

    orthogonal_criterion = covariance_orthogonal_loss
    classification_criterion = nn.CrossEntropyLoss().to(device)
    contrastive_criterion = domain_contrastive_loss
    mse_criterion = nn.MSELoss().to(device)

    # create combined dataloader
    min_len = min(len(source_train_loader), len(target_train_loader))
    combined_loader = zip(
        iter(source_train_loader),
        iter(target_train_loader))

    # 训练记录
    train_losses = {
        'total': [], 'class': [], 'orthogonal': [], 'contrastive': []
    }

    best_acc = 0
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
            (class_logits, domain_logits, features_class,
             features_domain, features_domain_sum, recons, domain_logits_from_upper) = model(combined_data)

            # calculate modulation classification loss (only source-domain)
            source_class_logits = class_logits[:batch_size]

            # 分类损失
            class_loss = classification_criterion(source_class_logits, source_labels)
            # 域分类损失
            # domain_upper_loss = classification_criterion(domain_logits_from_upper, combined_domains)
            # 域对抗损失
            domain_adversial_loss = classification_criterion(domain_logits, combined_domains)
            # 特征正交损失
            orthogonal_loss = orthogonal_criterion(torch.sum(features_class, dim=1),
                                                   torch.sum(features_domain, dim=1))
            # orthogonal_loss = covariance_orthogonal_loss_3d(features_class, features_domain)
            # 对比损失
            contrastive_loss = contrastive_criterion(features_domain_sum)
            # 重建损失
            recons_loss = mse_criterion(combined_data, recons)

            total_loss \
                = (class_loss
                   + domain_adversial_loss
                   + 0.5 * orthogonal_loss
                   + 0.5 * contrastive_loss
                   + 0.5 * recons_loss)

            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['class'] += class_loss.item()
            epoch_losses['orthogonal'] += orthogonal_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()

            # 每50个batch打印一次进度
            if batch_idx % 20 == 0:
                print(f'[Step2]Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{min_len}], '
                      f'Total Loss: {total_loss.item():.4f}, Class Loss: {class_loss.item():.4f}, '
                      f'domain adversial Loss: {domain_adversial_loss.item():.4f}, '
                      f'Orthogonal Loss: {orthogonal_loss.item():.4f}, '
                      f'Contrastive Loss: {contrastive_loss.item():.4f}'
                      f'Reconstruct Loss: {recons_loss.item():.4f}')

        for key in epoch_losses:
            epoch_losses[key] /= min_len
            train_losses[key].append(epoch_losses[key])

        acc = valid_accuracy = validate_model(model, target_train_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"../autodl-tmp/uda/{model_name}/" + f'{model_name}_{seq}.pth')

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

            class_logits, _, _, _, _, _, _ = model(data)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy


if __name__ == "__main__":
    for i in range(5):
        run_train("distan_dann", i)
