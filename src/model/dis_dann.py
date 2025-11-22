import torch
from torch import nn


class DisDANN(nn.Module):
    def __init__(self):
        super(DisDANN, self).__init__()

        self.fe_domain = nn.Sequential(
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

        self.fe_class = nn.Sequential(
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

        self.classifier = nn.Sequential(
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

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=8192, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=2048, out_features=11))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        feature_domain = self.fe_domain(x)
        feature_domain = feature_domain.view(feature_domain.size(0), -1)

        feature_class = self.fe_class(x)
        feature_class = feature_class.view(feature_class.size(0), -1)
        feature_class -= feature_domain

        # domain classification
        domain_logits = self.domain_classifier(feature_domain)

        # class classification
        class_logits = self.classifier(feature_class)

        return class_logits, domain_logits, feature_domain, feature_class


class dann_wrapper(nn.Module):
    def __init__(self, dann):
        super(dann_wrapper, self).__init__()

        self.dann = dann

    def forward(self, x):
        class_logits, _ = self.dann(x, 1.0)
        return class_logits


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DisDANN()

    sgn, _, _ = net(sgn)

    print(sgn.shape)