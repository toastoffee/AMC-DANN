import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import warnings

from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from module.mcldnn import MCLDNN
from train.train_classics import eval_and_get_acc

warnings.filterwarnings('ignore')


def run_train():
    device: torch.device = get_device()

    batch_size = 512
    num_epochs = 50

    train_source_loader, _ = DataloaderHelper.dataloader_10a(batch_size, 1.0)
    train_target_loader, _ = DataloaderHelper.dataloader_22(batch_size, 1.0)

    model = MCLDNN(num_classes=11)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        scheduler.step()

        for batch_idx, (data, labels, snr) in enumerate(train_source_loader):

            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device)

            out = model(data)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_source_loader)}], '
                      f'Loss: {loss.item():.4f}, ')

        if epoch % 5 == 0:
            eval_and_get_acc(model, train_target_loader, criterion, device, "[target]")


if __name__ == "__main__":
    run_train()
