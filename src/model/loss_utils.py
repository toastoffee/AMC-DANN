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


def covariance_orthogonal_loss_3d(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    对形状为 [B, N, M] 的两个特征张量计算协方差正交损失。
    内部调用原始的 2D 版本 `covariance_orthogonal_loss`。

    Args:
        z1: [B, N, M]
        z2: [B, N, M]

    Returns:
        正交损失标量
    """
    assert z1.dim() == 3 and z2.dim() == 3, "输入必须是3D张量 [B, N, M]"
    assert z1.shape == z2.shape, f"z1 和 z2 形状必须相同，但得到 {z1.shape} vs {z2.shape}"

    B, N, M = z1.shape
    # 将 [B, N, M] reshape 为 [B*N, M]，视为“大 batch”
    z1_2d = z1.view(-1, M)  # [B*N, M]
    z2_2d = z2.view(-1, M)  # [B*N, M]

    # 直接调用你原来的函数
    return covariance_orthogonal_loss(z1_2d, z2_2d)


def domain_contrastive_loss(features, temperature=0.1, eps=1e-8):
    """
    Domain-aware contrastive loss:
      - Pull samples from the same domain together
      - Push samples from different domains apart

    Args:
        features: [2N, D] — first N: source, last N: target
        temperature: float > 0
        eps: small constant for numerical stability

    Returns:
        loss: scalar tensor
    """
    device = features.device
    N2 = features.size(0)
    assert N2 % 2 == 0, "Batch size must be even (N source + N target)"
    N = N2 // 2

    # L2 normalize features (critical for cosine similarity)
    features = F.normalize(features, p=2, dim=1)  # [2N, D]

    # Compute cosine similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # [2N, 2N]

    # Build masks
    # Same-domain mask (excluding self)
    same_domain_mask = torch.zeros(N2, N2, dtype=torch.bool, device=device)
    same_domain_mask[:N, :N] = True  # source-source
    same_domain_mask[N:, N:] = True  # target-target
    same_domain_mask.fill_diagonal_(False)  # exclude self

    # Cross-domain mask (all source<->target pairs)
    cross_domain_mask = ~same_domain_mask.clone()
    cross_domain_mask.fill_diagonal_(False)  # though already false

    # Numerically stable log-softmax style computation
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()

    exp_sim = torch.exp(sim_stable)

    # Positive sum: same domain
    pos_sum = (exp_sim * same_domain_mask.float()).sum(dim=1)  # [2N]

    # Negative sum: cross domain
    neg_sum = (exp_sim * cross_domain_mask.float()).sum(dim=1)  # [2N]

    # Avoid division by zero
    denominator = pos_sum + neg_sum + eps
    loss_per_sample = -torch.log((pos_sum + eps) / denominator)

    return loss_per_sample.mean()