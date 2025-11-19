import torch
from torch import nn
import torch.nn.functional as F


def domain_aware_contrastive_loss(feature_domain, domain_labels, contrastive_loss_fn, source_batch_size):
    """
    🎯 基于域标签的对比学习损失函数
    目标：相同域的样本特征靠近，不同域的样本特征远离

    Args:
        feature_domain: 域特征 [2*batch_size, feature_dim]
        domain_labels: 域标签 [2*batch_size]
        contrastive_loss_fn: InfoNCE损失函数
        source_batch_size: 源域batch大小
    """
    total_batch_size = feature_domain.size(0)
    target_batch_size = total_batch_size - source_batch_size

    # 🎯 分离源域和目标域特征
    source_domain_features = feature_domain[:source_batch_size]  # [batch_size, feature_dim]
    target_domain_features = feature_domain[source_batch_size:]  # [batch_size, feature_dim]

    # 🎯 创建对比学习样本对
    # 策略1: 源域内部对比（源域样本间）
    source_contrastive_loss = intra_domain_contrastive_loss(
        source_domain_features, domain_labels[:source_batch_size], contrastive_loss_fn
    )

    # 策略2: 目标域内部对比（目标域样本间）
    target_contrastive_loss = intra_domain_contrastive_loss(
        target_domain_features, domain_labels[source_batch_size:], contrastive_loss_fn
    )

    # 策略3: 跨域对比（源域vs目标域）
    cross_domain_loss = inter_domain_contrastive_loss(
        source_domain_features, target_domain_features, contrastive_loss_fn
    )

    # 🎯 组合不同对比损失
    contrastive_loss = (
                               source_contrastive_loss +
                               target_contrastive_loss +
                               cross_domain_loss
                       ) / 3.0  # 平均权重

    return contrastive_loss


def intra_domain_contrastive_loss(domain_features, domain_labels, contrastive_loss_fn):
    """
    🎯 域内对比学习：相同域内的样本应该特征相似
    """
    batch_size = domain_features.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=domain_features.device)

    # 归一化特征
    domain_features = F.normalize(domain_features, dim=1)

    # 创建正负样本对
    # 正样本：同域内的其他样本
    # 负样本：由于是域内对比，所有样本都来自同一域，所以需要特殊处理

    # 使用所有其他样本作为负样本
    similarity_matrix = torch.mm(domain_features, domain_features.t())  # [batch_size, batch_size]

    # 创建标签：对角线为1（正样本），其他为0（负样本）
    labels = torch.eye(batch_size, device=domain_features.device)

    # 计算对比损失
    loss = contrastive_loss_fn(similarity_matrix, similarity_matrix, labels)

    return loss


def inter_domain_contrastive_loss(source_features, target_features, contrastive_loss_fn):
    """
    🎯 跨域对比学习：不同域的样本应该特征远离
    """
    # 归一化特征
    source_features = F.normalize(source_features, dim=1)
    target_features = F.normalize(target_features, dim=1)

    # 计算源域和目标域样本间的相似度
    cross_similarity = torch.mm(source_features, target_features.t())  # [batch_size, batch_size]

    # 创建标签：所有跨域样本对都是负样本（标签为0）
    labels = torch.zeros(cross_similarity.size(0), cross_similarity.size(1),
                         device=source_features.device)

    # 计算对比损失
    loss = contrastive_loss_fn(cross_similarity, cross_similarity, labels)

    return loss


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]