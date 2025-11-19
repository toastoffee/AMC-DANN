import torch


def mix_up(a: torch.Tensor,
           b: torch.Tensor,
           a_ratio: float,
           b_ratio: float):
    return a * a_ratio + b * b_ratio
