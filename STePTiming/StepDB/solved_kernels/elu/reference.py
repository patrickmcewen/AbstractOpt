"""PyTorch reference: ELU activation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims.get("alpha", 1.0)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
