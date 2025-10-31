import torch
from torch import nn


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU())

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32768, out_features=2048),
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
        feature = self.feature_extractor(x)
        feature = feature.view(feature.size(0), -1)

        logits = self.classifier(feature)

        return logits


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = CNN1d()

    sgn = net(sgn)

    print(sgn.shape)