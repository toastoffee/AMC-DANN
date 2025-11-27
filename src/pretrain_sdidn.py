import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from module.sdidn import DISTAN_G, HighFidelitySignalDecoder
from train.mask_pretrain import apply_random_mask, reconstruction_loss

warnings.filterwarnings('ignore')


def run_train(da_dataset: str):
    device: torch.device = get_device()

    batch_size = 512
    num_epochs = 50

    loader_10a, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    loader_22, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)
    min_len = min(len(loader_10a), len(loader_22))

    encoder = DISTAN_G().to(device)
    decoder = HighFidelitySignalDecoder().to(device)
    criterion = reconstruction_loss

    # optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ], lr=1e-3, weight_decay=1e-4)

    encoder.train()
    decoder.train()

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
            features = encoder(x_masked)
            x_recon = decoder(features)

            loss = criterion(x_recon, x, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(encoder.state_dict(), f"../autodl-tmp/uda/{da_dataset}/sdidn_pretrain/" + f'pretrained_sdidn_encoder.pth')
                torch.save(decoder.state_dict(), f"../autodl-tmp/uda/{da_dataset}/sdidn_pretrain/" + f'pretrained_sdidn_decoder.pth')
                print(f"new best loss: {loss.item()}, weights saved")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_train("16a_22")
