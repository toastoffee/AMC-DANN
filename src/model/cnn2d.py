import torch
from torch import nn
import torch.nn.functional as F

class FixedPadding(nn.Module):
    """可集成到Sequential中的填充层"""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding  # (left, right, top, bottom)

    def forward(self, x):
        return F.pad(x, self.padding, mode='constant', value=0)

    def __repr__(self):
        return f"FixedPadding(padding={self.padding})"


class CNN2d(nn.Module):
    def __init__(self, dropout_rate: float = 0.5):
        super(CNN2d, self).__init__()

        self.feature_extractor = nn.Sequential(
            FixedPadding((0, 7, 0, 1)),
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 8),
                      stride=1, padding=(0, 0), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 时间维度减半
            nn.Dropout(dropout_rate),

            FixedPadding((0, 7, 0, 1)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 8),
                      stride=1, padding=(0, 0), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),

            FixedPadding((0, 7, 0, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 8),
                      stride=1, padding=(0, 0), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),

            FixedPadding((0, 7, 0, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 8),
                      stride=1, padding=(0, 0), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),

            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=11),
            nn.Softmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        feature = self.feature_extractor(x)

        logits = self.classifier(feature)

        return logits


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = CNN2d()

    sgn = net(sgn)

    print(sgn.shape)
