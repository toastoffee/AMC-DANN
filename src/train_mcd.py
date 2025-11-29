import torch
from torch import nn, optim
import warnings


import torch.nn.functional as F
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from model.mcd import MCD
from dataset.dataset_utils import set_seeds
from model import modelutils
from sklearn.metrics import accuracy_score, top_k_accuracy_score

warnings.filterwarnings('ignore')


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


def run_train(source_train_loader, target_train_loader, da_dataset: str, model_name: str, seq: int):
    device: torch.device = get_device()

    set_seeds(seq)

    num_epochs = 20
    num_k = 4

    model = MCD()
    model.to(device)
    model.train()

    opt_g = optim.Adam(model.generator.parameters(), lr=1e-3, weight_decay=5e-4)
    opt_c1 = optim.Adam(model.classifier1.parameters(), lr=1e-3, weight_decay=5e-4)
    opt_c2 = optim.Adam(model.classifier2.parameters(), lr=1e-3, weight_decay=5e-4)

    def reset_grad():
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

    criterion = nn.CrossEntropyLoss().to(device)

    min_len = min(len(source_train_loader), len(target_train_loader))

    best_acc = 0
    for epoch in range(num_epochs):

        combined_loader = zip(iter(source_train_loader), iter(target_train_loader))
        for batch_idx, ((source_data, source_labels, source_snr),
                        (target_data, target_labels, target_snr)) in enumerate(combined_loader):

            if batch_idx >= min_len:
                break

            data_s = source_data.to(device, dtype=torch.float32)
            label_s = source_labels.to(device)
            data_t = target_data.to(device, dtype=torch.float32)
            data = torch.cat([data_s, data_t], dim=0)

            # step 1: train G and C, minimizing classification loss
            out_s1, out_s2 = model(data_s)
            loss_s1 = criterion(out_s1, label_s)
            loss_s2 = criterion(out_s2, label_s)
            loss_s = loss_s1 + loss_s2

            reset_grad()
            loss_s.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()

            # step 2: freeze G, train classifier
            out_s1, out_s2 = model(data_s)
            out_t1, out_t2 = model(data_t)
            loss_s1 = criterion(out_s1, label_s)
            loss_s2 = criterion(out_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = discrepancy(out_t1, out_t2)
            loss = loss_s - loss_dis
            reset_grad()
            loss.backward()
            opt_c1.step()
            opt_c2.step()

            # step 3: train G
            for i in range(4):
                out_t1, out_t2 = model(data_t)
                loss_dis = discrepancy(out_t1, out_t2)
                reset_grad()
                loss_dis.backward()
                opt_g.step()

            if batch_idx % 50 == 0:
                print(f'[Step2]Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{min_len}], '
                      f'Loss s1: {loss_s1.item():.4f}, '
                      f'Loss s2: {loss_s2.item():.4f}, '
                      f'Discrepancy: {loss_dis.item():.4f}, ')

        acc = validate_model(model, target_train_loader, device)
        if acc > best_acc:
            best_acc = acc
            print(f"new best acc:{best_acc}, weights saved")
            torch.save(model.state_dict(),
                       f"../autodl-tmp/uda/{da_dataset}/{model_name}/" + f'{model_name}_{seq}.pth')


def validate_model(model, valid_loader, device):
    """
    🎯 验证模型在源域上的性能
    """
    model.eval()
    correct1 = 0
    correct2 = 0
    total = 0

    with torch.no_grad():
        for data, labels, snr in valid_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            class_logits1, class_logits2 = model(data)
            _, predicted1 = torch.max(class_logits1.data, 1)
            _, predicted2 = torch.max(class_logits2.data, 1)

            total += labels.size(0)
            correct1 += (predicted1 == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()

    accuracy1 = 100 * correct1 / total
    accuracy2 = 100 * correct2 / total

    print(f"acc1:{accuracy1}, acc2:{accuracy2}")

    model.train()  # 恢复训练模式

    return accuracy1


if __name__ == "__main__":

    batch_size = 512

    # source_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    # target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    # for i in range(3):
    #     run_train(source_train_loader, target_train_loader, "16a_22", "mcd", i)

    # 22->16a
    source_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "22_16a", "mcd", i)

    # 16c->22
    source_train_loader, _ = DataloaderHelper.dataloader_04c(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "16c_22", "mcd", i)

    # 22->16c
    source_train_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    target_train_loader, _ = DataloaderHelper.dataloader_04c(batch_size, 1.0)
    for i in range(3):
        run_train(source_train_loader, target_train_loader, "22_16c", "mcd", i)
