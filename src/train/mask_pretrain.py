import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def apply_random_mask(x, mask_ratio=0.4):
    """
    x: [B, C, L]  e.g., [B, 2, 128]
    Returns:
        x_masked: input with masked positions set to 0 (or noise)
        mask: [B, L], 1 where masked, 0 otherwise
    """
    B, C, L = x.shape
    num_mask = int(L * mask_ratio)

    # Randomly choose indices to mask for each sample
    mask = torch.zeros(B, L, device=x.device)
    idx = torch.randperm(L)
    mask[:, idx[:num_mask]] = 1  # shape [B, L]

    # Apply mask: set masked positions to 0
    x_masked = x.clone()
    x_masked = x_masked * (1 - mask.unsqueeze(1))  # broadcast over C

    return x_masked, mask.bool()


def apply_blockwise_mask(x, mask_ratio=0.4, block_size=8):
    """
    Apply block-wise random masking to 1D sequence.

    Args:
        x: input tensor of shape [B, C, L]
        mask_ratio: ratio of total positions to mask (e.g., 0.4 = 40%)
        block_size: length of each masked block (default: 8)

    Returns:
        x_masked: masked input, same shape as x
        mask: boolean tensor of shape [B, L], True where masked
    """
    B, C, L = x.shape
    device = x.device

    # Total number of tokens to mask
    len_mask = int(L * mask_ratio)
    if len_mask == 0:
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        return x.clone(), mask

    # Number of complete blocks to mask
    num_blocks = len_mask // block_size
    if num_blocks == 0:
        # If mask_ratio too small, fall back to single-point masking
        return apply_random_mask(x, mask_ratio)

    # Randomly choose starting positions for blocks (ensure they fit in [0, L))
    max_start = L - block_size
    if max_start < 0:
        # block larger than sequence → mask entire sequence
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
        x_masked = torch.zeros_like(x)
        return x_masked, mask

    # Sample start indices uniformly
    start_indices = torch.randint(0, max_start + 1, (B, num_blocks), device=device)  # [B, num_blocks]

    # Create mask
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        for i in range(num_blocks):
            start = start_indices[b, i]
            mask[b, start:start + block_size] = True

    # Ensure we don't exceed desired mask ratio (optional clipping)
    # (Not strictly necessary, but keeps ratio close)
    actual_masked = mask.sum(dim=1).float().mean().item()
    # If you want exact ratio, you can trim extra, but usually not needed.

    # Apply mask
    x_masked = x.clone()
    x_masked = x_masked * (~mask.unsqueeze(1))  # broadcast over channel dim

    return x_masked, mask

def reconstruction_loss(pred, target, mask):
    """
    Compute MSE only on masked positions.
    pred, target: [B, C, L]
    mask: [B, L], bool or 0/1
    """
    # Only compute loss where mask == 1
    diff = (pred - target) ** 2  # [B, C, L]
    loss = diff * mask.unsqueeze(1)  # [B, C, L]
    return loss.sum() / (mask.sum() * pred.size(1) + 1e-8)  # normalize by #masked * channels