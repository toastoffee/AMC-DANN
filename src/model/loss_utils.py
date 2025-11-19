import torch
import torch.nn.functional as F


def covariance_orthogonal_loss(z1, z2):
    """
    计算基于协方差矩阵的正交损失。
    目标：让拼接特征Z=[z1, z2]的协方差矩阵接近单位矩阵。
    """
    # 拼接特征
    z = torch.cat([z1, z2], dim=1)  # 形状: [batch_size, z1_dim + z2_dim]
    batch_size, dim = z.size()

    # 中心化特征（减去均值）
    z_centered = z - z.mean(dim=0, keepdim=True)

    # 计算协方差矩阵: (Z^T * Z) / (batch_size - 1)
    cov = torch.mm(z_centered.t(), z_centered) / (batch_size - 1)  # 形状: [dim, dim]

    # 目标是对角矩阵（单位矩阵）
    identity = torch.eye(dim, device=z.device)

    # 计算Frobenius范数下的差异
    loss = F.mse_loss(cov, identity)
    return loss
