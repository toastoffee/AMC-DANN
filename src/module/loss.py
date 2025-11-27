import torch
from torch import nn
import torch.nn.functional as F

from utils import entropy
from kernels import GaussianKernel


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


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels, linear=False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix= None,
                         linear= True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
