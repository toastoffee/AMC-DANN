import torch
import torch.nn as nn
import torch.nn.functional as F

from grl import GradientReversalFunction

# NAME: Signal Domain-Invariant Disentangling Network
# SDIDN


class DistanClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super(DistanClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SELU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(inplace=True)
        )
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, features):
        """
        features: [B, 128]
        Returns: logits [B, num_classes]
        """
        x = self.fc1(features)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits


class DistanDANN(nn.Module):
    def __init__(self):
        super(DistanDANN, self).__init__()
        self.g = nn.Sequential(
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

        self.reconstructor = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=2,
                      kernel_size=3, padding=1, stride=1))

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=8192, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=256, out_features=2))

        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=8192, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=256, out_features=11))

        self.domain_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128))

        self.class_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128))

        self.mapping = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128))

    def forward(self, x, alpha):
        features = self.g(x)

        features_domain = self.domain_mlp(features)
        features_class = self.class_mlp(features)
        features_class_no_grad = self.class_mlp(features.detach())
        reversed_features_class_no_grad = GradientReversalFunction.apply(features_class_no_grad, alpha)

        features_all = features_domain + features_class
        features_all = self.mapping(features_all)
        recons = self.reconstructor(features_all)

        features_domain_sum = features_domain.view(features_domain.size(0), -1)
        features_class_sum = features_class.view(features_class.size(0), -1)
        reversed_features_class_no_grad_sum = reversed_features_class_no_grad.view(reversed_features_class_no_grad.size(0), -1)

        class_logits = self.class_classifier(features_class_sum)
        domain_logits = self.domain_classifier(reversed_features_class_no_grad_sum)
        domain_logits_from_upper = self.domain_classifier(features_domain_sum)

        return class_logits, domain_logits, features_class, features_domain, features_domain_sum, recons, domain_logits_from_upper


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DistanDANN()

    p = net(sgn, 1.0)

    print(p[1].shape)