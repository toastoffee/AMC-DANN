import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = out[:, -1, :]  # 取最后一个时间步
        return out


class MCLDNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(MCLDNNFeatureExtractor, self).__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True)
        self.conv2 = nn.Conv1d(1, 50, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv1d(1, 50, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True)
        self.conv5 = nn.Conv2d(100, 100, kernel_size=(2, 5), stride=1, bias=True)

        # LSTM encoder
        self.encoder_layer_t = LSTMEncoder(input_size=100, hidden_size=128)

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

        features = self.encoder_layer_t(y)  # [B, 128]
        return features


class MCLDNNClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super(MCLDNNClassifier, self).__init__()
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


class MCLDNN(nn.Module):
    def __init__(self, num_classes):
        super(MCLDNN, self).__init__()
        self.feature_extractor = MCLDNNFeatureExtractor()
        self.classifier = MCLDNNClassifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
