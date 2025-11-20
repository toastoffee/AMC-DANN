import torch
from torch import nn


class DualDANN(nn.Module):
    def __init__(self):
        super(DualDANN, self).__init__()

        self.feature_extractor = nn.Sequential(
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

        self.domain_mlp = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.6))

        self.class_mlp = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.6))

        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=11),
            nn.Softmax(dim=1))

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        shallow_features = self.feature_extractor(x)
        shallow_features = shallow_features.view(shallow_features.size(0), -1)

        # domain feature
        feature_domain = self.domain_mlp(shallow_features)
        feature_class = self.class_mlp(shallow_features)
        # feature_class -= feature_domain

        # domain classification
        domain_logits = self.domain_classifier(feature_domain)

        # class classification
        class_logits = self.class_classifier(feature_class)

        return class_logits, domain_logits, feature_domain, feature_class


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DualDANN()

    sgn, _, _, _ = net(sgn)

    print(sgn.shape)