
import torch
from torch import nn
import torch.nn.functional as F


class DAN(nn.Module):
    def __init__(self, num_classes=11, hidden_size=128):
        super(DAN, self).__init__()
        # 特征提取器：CNN + LSTM
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(32)  # 固定长度

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True)

        # 分类器（仅用于源域）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # 记录用于 MMD 的层输出
        self.adapt_layers = ['conv_out', 'lstm_out']  # 可扩展

    def forward(self, x):
        """
        x: [B, 2, 128]
        Returns: logits, features_dict
        """
        # CNN
        x = torch.relu(self.conv1(x))  # [B, 64, 128]
        x = torch.relu(self.conv2(x))  # [B, 128, 128]
        x = torch.relu(self.conv3(x))  # [B, 128, 128]
        conv_out = self.pool(x)  # [B, 128, 32]

        # LSTM
        x = conv_out.transpose(1, 2)  # [B, 32, 128]
        lstm_out, _ = self.lstm(x)  # [B, 32, 128]
        lstm_out = lstm_out[:, -1, :]  # [B, 128] 取最后一步

        # 分类
        logits = self.classifier(lstm_out)

        # 返回中间特征用于 MMD
        features = {
            'conv_out': conv_out.view(conv_out.size(0), -1),  # [B, 128*32]
            'lstm_out': lstm_out  # [B, 128]
        }
        return logits, features


class ImageClassifier(nn.Module):
    """
    兼容 DAN 训练脚本的信号分类器。
    输入: [B, 2, 128]  (I/Q)
    输出: (logits, features)
        - logits: [B, num_classes]
        - features: [B, feature_dim]  # 用于 MK-MMD 的域对齐特征
    """

    def __init__(self, num_classes=11, backbone=None, pool_size=16, hidden_dim=128):
        super(ImageClassifier, self).__init__()

        # 如果未提供自定义 backbone，则使用默认 CNN
        if backbone is None:
            self.backbone = SignalBackbone(pool_size=pool_size, hidden_dim=hidden_dim)
        else:
            self.backbone = backbone

        # 分类头（不参与 MMD，仅用于分类）
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 2, 128]
        Returns:
            logits: [B, num_classes]
            features: [B, hidden_dim]  ← 这个会被送入 mkmmd_loss
        """
        features = self.backbone(x)  # [B, hidden_dim]
        logits = self.head(features)  # [B, num_classes]
        return logits, features


class SignalBackbone(nn.Module):
    """特征提取主干网络"""

    def __init__(self, pool_size=16, hidden_dim=128):
        super(SignalBackbone, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)  # [B, 128, pool_size]

        # 将池化后的特征展平并通过 FC 得到最终特征向量
        self.fc = nn.Linear(128 * pool_size, hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # [B, 128, P]
        x = x.view(x.size(0), -1)  # [B, 128*P]
        x = self.fc(x)  # [B, hidden_dim]
        return x


class DAN_wrapper(nn.Module):
    def __init__(self, dan: DAN):
        super(DAN_wrapper, self).__init__()
        self.dan = dan

    def forward(self, x):
        return self.dan(x)[0]


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = ImageClassifier()

    sgn, _ = net(sgn)

    print(sgn.shape)
    print(_.shape)