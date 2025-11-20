import torch
import torch.nn.functional as F


def covariance_orthogonal_loss(z1, z2):
    """
    特征间协方差正交损失 - 约束两个特征集之间的正交性

    Args:
        z1: 第一个特征提取器的输出，形状(batch_size, feature_dim)
        z2: 第二个特征提取器的输出，形状(batch_size, feature_dim)
        lambda_orth: 损失权重系数

    Returns:
        orth_loss: 正交损失值
    """
    assert z1.shape == z2.shape, "z1和z2的形状必须相同"

    batch_size, feature_dim = z1.shape

    # 特征中心化
    z1_centered = z1 - z1.mean(dim=0, keepdim=True)
    z2_centered = z2 - z2.mean(dim=0, keepdim=True)

    # 计算z1和z2之间的互协方差矩阵
    cross_cov = torch.mm(z1_centered.t(), z2_centered) / (batch_size - 1)

    # 计算互协方差矩阵的Frobenius范数
    orth_loss = torch.norm(cross_cov, p='fro')

    return orth_loss
