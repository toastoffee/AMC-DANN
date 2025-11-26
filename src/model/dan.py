
import torch
from torch import nn

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


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DAN()

    sgn, _ = net(sgn)

    print(sgn.shape)