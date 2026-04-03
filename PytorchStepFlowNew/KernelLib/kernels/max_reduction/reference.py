"""PyTorch reference: Max reduction along a dimension."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, reduce_dim=-1):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x):
        return torch.max(x, dim=self.reduce_dim)[0]


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims.get("reduce_dim", -1)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
