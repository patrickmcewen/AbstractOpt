"""PyTorch reference: Group normalization."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, C, G):
        super().__init__()
        self.gn = nn.GroupNorm(G, C)

    def forward(self, x):
        return self.gn(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, C, H, W = dims["B"], dims["C"], dims["H"], dims["W"]
    return [torch.randn(B, C, H, W)]


def get_init_inputs(dims):
    return [dims["C"], dims["G"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
