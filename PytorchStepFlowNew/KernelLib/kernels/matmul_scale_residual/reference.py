"""PyTorch reference: Matmul followed by scaling and residual addition."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.linear = nn.Linear(K, N, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out + out * 0.5


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims["K"], dims["N"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
