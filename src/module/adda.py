import torch
from torch import nn, optim
import torch.nn.functional as F
from train.device_utils import get_device
from model import modelutils


class ADDA(nn.Module):
    """
    Adversarial Discriminative Domain Adaptation 模型
    输入形状: [B, 2, 128] - I/Q信号数据
    """

    def __init__(self, num_classes: int = 11):
        super(ADDA, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = 256

        # 源编码器
        self.source_encoder = self._build_encoder()
        # 目标编码器 - 结构相同但参数独立
        self.target_encoder = self._build_encoder()
        # 分类器 - 源域和目标域共享
        self.classifier = self._build_classifier()
        # 域判别器
        self.discriminator = self._build_discriminator()

        # 初始化目标编码器权重（复制源编码器）
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())

        # 冻结目标编码器初始状态
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _build_encoder(self) -> nn.Module:
        """构建I/Q信号编码器"""
        return nn.Sequential(
            # [B, 2, 128] -> [B, 64, 124]
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [B, 64, 62]

            # [B, 64, 62] -> [B, 128, 58]
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [B, 128, 31]

            # [B, 128, 31] -> [B, 256, 27]
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # [B, 256, 1]

            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def _build_classifier(self) -> nn.Module:
        """构建分类器"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def _build_discriminator(self) -> nn.Module:
        """构建域判别器"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor, domain: str = 'source', return_features: bool = False):
        """
        前向传播
        Args:
            x: 输入信号 [B, 2, 128]
            domain: 'source' 或 'target'
            return_features: 是否返回特征
        """
        if domain == 'source':
            features = self.source_encoder(x)
        else:
            features = self.target_encoder(x)

        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_domain_prediction(self, features: torch.Tensor) -> torch.Tensor:
        """获取域判别结果"""
        return self.discriminator(features)

