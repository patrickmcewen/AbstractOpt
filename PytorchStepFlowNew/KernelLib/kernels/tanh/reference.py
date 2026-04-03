"""PyTorch reference: Tanh activation."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
