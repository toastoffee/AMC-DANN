import torch
from torch import nn


class MCD(nn.Module):
    def __init__(self):
        super(MCD, self).__init__()

        self.generator = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU())

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=256, out_features=11))

        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=256, out_features=11))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        features = self.generator(x)
        features = features.view(features.size(0), -1)

        # domain classification
        class_logits1 = self.classifier1(features)
        class_logits2 = self.classifier2(features)

        return class_logits1, class_logits2


class MCD_wrapper(nn.Module):
    def __init__(self, mcd: MCD):
        super(MCD_wrapper, self).__init__()
        self.mcd = mcd

    def forward(self, x):
        return self.mcd(x)[0]

if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = MCD()

    p1, p2 = net(sgn)

    print(p1.shape)