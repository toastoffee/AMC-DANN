import torch
import torch.nn as nn
import torch.nn.functional as F


class ADDA(nn.Module):
    """
    Adversarial Discriminative Domain Adaptation (ADDA) 模型。

    包含三个子模块：
        - source_encoder: 源域编码器（预训练后固定）
        - classifier: 分类器（预训练后固定）
        - target_encoder: 目标域编码器（对抗训练阶段更新）
        - domain_discriminator: 域判别器（对抗训练阶段更新）

    输入格式: [batch_size, 2, 128]
    """

    def __init__(self, num_classes=11, hidden_dim=128):
        super(ADDA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 共享结构的编码器（用于初始化）
        self._encoder_template = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, hidden_dim)
        )

        # 源域编码器（预训练 + 固定）
        self.source_encoder = nn.Sequential(*list(self._encoder_template.children()))

        # 目标域编码器（从源复制，对抗训练时更新）
        self.target_encoder = nn.Sequential(*list(self._encoder_template.children()))

        # 分类器（仅在源域预训练，之后固定）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # 域判别器（对抗训练）
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_source(self, x):
        """用于第一阶段：源域预训练"""
        feat = self.source_encoder(x)
        logits = self.classifier(feat)
        return logits, feat

    def forward_target(self, x):
        """用于第二阶段：目标域特征提取"""
        return self.target_encoder(x)

    def discriminate(self, feat):
        """域判别"""
        return self.domain_discriminator(feat).squeeze(-1)  # [B]

    def copy_source_to_target(self):
        """将源编码器权重复制给目标编码器（第二阶段开始前调用）"""
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())

    def freeze_source_and_classifier(self):
        """冻结源编码器和分类器（第二阶段使用）"""
        for param in self.source_encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_target_and_disc(self):
        """确保目标编码器和判别器可训练"""
        for param in self.target_encoder.parameters():
            param.requires_grad = True
        for param in self.domain_discriminator.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = ADDA()

    sgn, _ = net.forward_source(sgn)

    print(sgn.shape)
    print(_.shape)
