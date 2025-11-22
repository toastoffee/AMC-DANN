import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from log_utils import log_info


def train_epoch_dann(model:           nn.Module,
                     source_loader:    DataLoader,
                     target_loader:    DataLoader,
                     epoch:            int,
                     num_epochs:       int,
                     optimizer:        optim.Optimizer,
                     class_criterion:  nn.Module,
                     domain_criterion: nn.Module,
                     device:           torch.device,
                     header_desc:      str):
    # set the model to training mode
    model.train()
    total_class_loss = 0.0
    total_domain_loss = 0.0
    source_correct = 0
    source_total = 0

    # create combined dataloader
    min_len = min(len(source_loader), len(target_loader))
    combined_loader = zip(
        iter(source_loader),
        iter(target_loader))

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

        # calculate alpha dynamicly
        p = (epoch * min_len + batch_idx) / (num_epochs * min_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1.

        # forward
        class_logits, domain_logits = model(combined_data, alpha)

        # calculate modulation classification loss (only source-domain)
        source_class_logits = class_logits[:batch_size]
        class_loss = class_criterion(source_class_logits, source_labels)

        # calculate domain loss
        domain_loss = domain_criterion(domain_logits, combined_domains)

        total_loss = class_loss + domain_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 统计信息
        total_class_loss += class_loss.item()
        total_domain_loss += domain_loss.item()

        # 计算源域准确率
        source_class_preds = torch.argmax(source_class_logits, dim=1)
        source_acc = accuracy_score(source_class_preds.cpu(), source_labels.cpu())
        source_correct += source_acc * batch_size
        source_total += batch_size

        print(f'Epoch: {epoch + 1}/{num_epochs} | '
              f'Batch: {batch_idx}/{min_len} | '
              f'Class Loss: {class_loss.item():.4f} | '
              f'Domain Loss: {domain_loss.item():.4f} | '
              f'Source Acc: {source_acc:.3f}')

    avg_class_loss = total_class_loss / min_len
    avg_domain_loss = total_domain_loss / min_len
    avg_source_acc = source_correct / source_total if source_total > 0 else 0.0

    return avg_class_loss, avg_domain_loss, avg_source_acc


def train_epoch_dann_alt(model:           nn.Module,
                         source_loader:    DataLoader,
                         target_loader:    DataLoader,
                         epoch:            int,
                         num_epochs:       int,
                         optimizer:        optim.Optimizer,
                         class_criterion:  nn.Module,
                         domain_criterion: nn.Module,
                         device:           torch.device,
                         header_desc:      str):
    # set the model to training mode
    model.train()
    total_class_loss = 0.0
    total_domain_loss = 0.0
    source_correct = 0
    source_total = 0

    # create combined dataloader
    min_len = min(len(source_loader), len(target_loader))
    combined_loader = zip(
        iter(source_loader),
        iter(target_loader))

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

        # calculate alpha dynamicly
        p = (epoch * min_len + batch_idx) / (num_epochs * min_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1.

        # train with source-domain data
        class_s_logits, domain_s_logits = model(source_data, alpha)
        class_s_loss = class_criterion(class_s_logits, source_labels)
        domain_s_loss = domain_criterion(domain_s_logits, source_domains)

        # train with target-domain data
        _, domain_t_logits = model(target_data, alpha)
        domain_t_loss = domain_criterion(domain_t_logits, target_domains)

        total_loss = class_s_loss + domain_s_loss + domain_t_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 统计信息
        total_class_loss += class_s_loss.item()
        total_domain_loss += domain_s_loss.item() + domain_t_loss.item()

        # 计算源域准确率
        source_class_preds = torch.argmax(class_s_logits, dim=1)
        source_acc = accuracy_score(source_class_preds.cpu(), source_labels.cpu())
        source_correct += source_acc * batch_size
        source_total += batch_size

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch + 1}/{num_epochs} | '
                  f'Batch: {batch_idx}/{min_len} | '
                  f'Class Loss: {class_s_loss.item():.4f} | '
                  f'Domain Loss: {(domain_s_loss.item() + domain_t_loss.item()):.4f} | '
                  f'Source Acc: {source_acc:.3f}')

    avg_class_loss = total_class_loss / min_len
    avg_domain_loss = total_domain_loss / min_len
    avg_source_acc = source_correct / source_total if source_total > 0 else 0.0

    return avg_class_loss, avg_domain_loss, avg_source_acc


def train_dann(model:            nn.Module,
               source_loader:    DataLoader,
               target_loader:    DataLoader,
               target_valid_loader: DataLoader,
               optimizer:        optim.Optimizer,
               device:           torch.device,
               num_epochs:       int,
               model_name:       str):
    log_info("start training: " + model_name)

    model.to(device)
    class_criterion = nn.CrossEntropyLoss().to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        train_epoch_dann_alt(model, source_loader, target_loader,
                         epoch, num_epochs, optimizer,
                         class_criterion, domain_criterion, device, "placeholder")

        valid_accuracy = validate_model(model, target_valid_loader, device)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {valid_accuracy:.2f}%')

    torch.save(model.state_dict(), model_name + '.pth')


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

            class_logits, _ = model(data, 0.0)
            _, predicted = torch.max(class_logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()  # 恢复训练模式

    return accuracy
