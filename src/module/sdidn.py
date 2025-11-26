import torch
import torch.nn as nn
import torch.nn.functional as F

from grl import GradientReversalFunction

# NAME: Signal Domain-Invariant Disentangling Network
# SDIDN


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        # out = out[:, -1, :]  # 取最后一个时间步
        return out


class DISTAN_G(nn.Module):
    def __init__(self):
        super(DISTAN_G, self).__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True)
        self.conv2 = nn.Conv1d(1, 50, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv1d(1, 50, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True)
        self.conv5 = nn.Conv2d(100, 100, kernel_size=(2, 5), stride=1, bias=True)
        self.lstm = LSTMEncoder(input_size=100, hidden_size=128)

        # 初始化权重（可选）
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [B, 2, L]  e.g., [B, 2, 128]
        Returns: features of shape [B, 128]
        """
        y = x.unsqueeze(1)  # [B, 1, 2, L]

        I = y[:, :, 0, :]   # [B, 1, L]
        Q = y[:, :, 1, :]   # [B, 1, L]

        y = self.conv1(y)   # [B, 50, 2, L]

        I = self.conv2(I).unsqueeze(2)  # [B, 50, 1, L]
        Q = self.conv3(Q).unsqueeze(2)  # [B, 50, 1, L]

        IQ = torch.cat((I, Q), dim=2)   # [B, 50, 2, L]
        IQ = self.conv4(IQ)             # [B, 50, 2, L]

        y = torch.cat((IQ, y), dim=1)   # [B, 100, 2, L]
        y = self.conv5(y)               # [B, 100, 1, L_out]  (L_out = L - 4 + 1 = L-3 if L=128 → 125)

        y = y.squeeze(2)                # [B, 100, L_out]
        y = y.transpose(1, 2)           # [B, L_out, 100]

        y = self.lstm(y)
        return y


class HighFidelitySignalDecoder(nn.Module):
    def __init__(self, input_dim=128, target_len=128, n_channels=2):
        super(HighFidelitySignalDecoder, self).__init__()
        self.target_len = target_len
        self.n_channels = n_channels

        # Optional: temporal modeling (LSTM helps if long-range dependency matters)
        self.use_lstm = True
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=input_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )

        # Channel projection
        self.proj = nn.Conv1d(input_dim, 128, kernel_size=1)

        # Multi-scale refinement with residual connections (U-Net style)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv1d(32, n_channels, kernel_size=1)

        # Optional: output activation (use only if your signal is normalized)
        # self.output_act = nn.Tanh()  # or nn.Identity()

    def forward(self, z):
        """
        z: [B, T, D] e.g., [B, 125, 128]
        Returns: [B, n_channels, target_len] e.g., [B, 2, 128]
        """
        B, T, D = z.shape

        # Optional: temporal refinement with LSTM
        if self.use_lstm:
            z, _ = self.lstm(z)  # [B, T, 128]

        # Transpose to [B, D, T] for Conv1D
        x = z.transpose(1, 2)  # [B, 128, T]

        # Project channels (identity if same dim)
        x = self.proj(x)  # [B, 128, T]

        # Upsample time dimension: T → target_len (e.g., 125 → 128)
        x = F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)

        # Refinement blocks
        x = self.conv_block1(x)   # [B, 64, 128]
        x = self.conv_block2(x)   # [B, 32, 128]
        x = self.out_conv(x)      # [B, 2, 128]

        # Optional: apply activation if signal is normalized
        # x = self.output_act(x)

        return x


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
        self.g = DISTAN_G()
        self.reconstructor = HighFidelitySignalDecoder()
        self.domain_classifier = DistanClassifier(num_classes=2)
        self.class_classifier = DistanClassifier(num_classes=11)

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

    def forward(self, x):
        features = self.g(x)

        features_domain = self.domain_mlp(features)
        features_class = self.class_mlp(features)
        features_class_no_grad = self.class_mlp(features.detach())
        reversed_features_class_no_grad = GradientReversalFunction.apply(features_class_no_grad, 1.0)

        features_all = features_domain + features_class
        features_all = self.mapping(features_all)
        recons = self.reconstructor(features_all)

        features_domain_sum = torch.sum(features_domain, dim=1)
        features_class_sum = torch.sum(features_class, dim=1)
        reversed_features_class_no_grad_sum = torch.sum(reversed_features_class_no_grad, dim=1)

        class_logits = self.class_classifier(features_class_sum)
        domain_logits = self.domain_classifier(reversed_features_class_no_grad_sum)
        domain_logits_from_upper = self.domain_classifier(features_domain_sum)

        return class_logits, domain_logits, features_class, features_domain, features_domain_sum, recons, domain_logits_from_upper


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DistanDANN()

    p = net(sgn)

    print(p[1].shape)