import torch
from torch import nn
from modelutils import GradientReversalFunction

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


class DANN(nn.Module):
    def __init__(self, grl_alpha=1.0):
        super(DANN, self).__init__()

        self.feature_extractor = DISTAN_G()

        self.classifier = DistanClassifier(num_classes=11)

        self.domain_classifier = DistanClassifier(num_classes=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, alpha: float):
        feature = self.feature_extractor(x)
        feature = torch.sum(feature, dim=1)

        reversed_features = GradientReversalFunction.apply(feature, alpha)

        # class classification
        class_logits = self.classifier(feature)
        # domain classification
        domain_logits = self.domain_classifier(reversed_features)

        return class_logits, domain_logits


class dann_wrapper(nn.Module):
    def __init__(self, dann):
        super(dann_wrapper, self).__init__()

        self.dann = dann

    def forward(self, x):
        class_logits, _ = self.dann(x, 1.0)
        return class_logits


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DANN()

    sgn, _ = net(sgn, 1.0)

    print(sgn.shape)
    print(_.shape)