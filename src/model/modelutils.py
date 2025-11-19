from torch.autograd import Function
import torch
from torch import nn
import torch.nn.functional as F


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def freeze(model: nn.Module):
    """
    freeze model params
    :param model: model to freeze
    :return:
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    """
    unfreeze model params
    :param model: model to unfreeze
    :return:
    """
    for param in model.parameters():
        param.requires_grad = True


