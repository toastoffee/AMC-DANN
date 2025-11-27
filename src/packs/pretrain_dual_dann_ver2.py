import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings
import os

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from model.dual_dann_ver2 import DualDANN
from train.mask_pretrain import apply_random_mask, reconstruction_loss

warnings.filterwarnings('ignore')


def run_train():
    device: torch.device = get_device()

    batch_size = 512
    num_epochs = 50

    loader_10a, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    loader_22, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    min_len = min(len(loader_10a), len(loader_22))

    model = DualDANN().to(device)
    criterion = reconstruction_loss

    # optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters()},
        {'params': model.reconstructor.parameters()}
    ], lr=1e-3, weight_decay=1e-4)

    model.train()

    # step2: train the entire network
    best_loss = 1000.0
    for epoch in range(num_epochs):

        combined_loader = zip(
            iter(loader_10a),
            iter(loader_22))
        for batch_idx, ((source_data, source_labels, source_snr),
                        (target_data, target_labels, target_snr)) in enumerate(combined_loader):

            if batch_idx >= min_len:
                break

            source_data = source_data.to(device, dtype=torch.float32)
            target_data = target_data.to(device, dtype=torch.float32)

            x = torch.cat([source_data, target_data], dim=0)

            x_masked, mask = apply_random_mask(x, mask_ratio=0.4)
            features = model.feature_extractor(x_masked)
            x_recon = model.reconstructor(features)

            loss = criterion(x_recon, x, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'pretrained_dual_dann.pth')
                print(f"new best loss: {loss.item()}, weights saved")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_train()
