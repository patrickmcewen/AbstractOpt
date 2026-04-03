"""PyTorch reference: Leaky ReLU activation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims.get("negative_slope", 0.01)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
