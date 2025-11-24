import torch

def build_domain_positive_pairs(feature_domain):
    """
    Build positive pairs for domain contrastive learning.

    Assumes:
        feature_domain: [2N, D]
        first N samples: source domain
        last N samples: target domain

    Returns:
        query: [2N, D]
        positive_key: [2N, D]  # each sample's positive is another from same domain
    """
    N = feature_domain.size(0) // 2

    # Split
    src_feat = feature_domain[:N]  # [N, D]
    tgt_feat = feature_domain[N:]  # [N, D]

    # Circular shift: x[i] -> x[(i+1) % N]
    src_pos = torch.cat([src_feat[1:], src_feat[:1]], dim=0)  # [N, D]
    tgt_pos = torch.cat([tgt_feat[1:], tgt_feat[:1]], dim=0)  # [N, D]

    # Combine
    query = feature_domain  # [2N, D]
    positive_key = torch.cat([src_pos, tgt_pos], dim=0)  # [2N, D]

    return query, positive_key


def build_domain_negative_pairs(feature_domain):
    """
    Build positive pairs for domain contrastive learning.

    Assumes:
        feature_domain: [2N, D]
        first N samples: source domain
        last N samples: target domain

    Returns:
        query: [2N, D]
        positive_key: [2N, D]  # each sample's positive is another from same domain
    """
    N = feature_domain.size(0) // 2

    # Split
    src_feat = feature_domain[:N]  # [N, D]
    tgt_feat = feature_domain[N:]  # [N, D]

    # Circular shift: x[i] -> x[(i+1) % N]
    src_pos = torch.cat([src_feat[N-1:], src_feat[:N-1]], dim=0)  # [N, D]
    tgt_pos = torch.cat([tgt_feat[N-1:], tgt_feat[:N-1]], dim=0)  # [N, D]

    # Combine
    query = feature_domain  # [2N, D]
    negative_key = torch.cat([tgt_pos, src_pos], dim=0)  # [2N, D]

    return query, negative_key


if __name__ == "__main__":
    feat = torch.randn(6, 4)  # N=3, D=4
    q, pos = build_domain_negative_pairs(feat)