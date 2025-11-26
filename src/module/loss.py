import torch
from torch import nn
import torch.nn.functional as F

from utils import entropy
from kernels import guassian_kernel


def CovarianceOrthogonalLoss(z1, z2):
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


def CovarianceOrthogonalLoss3D(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    协方差正交损失
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
    return CovarianceOrthogonalLoss(z1_2d, z2_2d)


def DomainContrastiveLoss(features, temperature=0.1, eps=1e-8):
    """
    域相关对比损失
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


class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


def mk_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算 MK-MMD 损失"""
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss