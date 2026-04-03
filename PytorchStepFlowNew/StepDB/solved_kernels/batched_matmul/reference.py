"""PyTorch reference: Batched matrix multiplication."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.bmm(A, B)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, M, K, N = dims["B"], dims["M"], dims["K"], dims["N"]
    return [torch.randn(B, M, K), torch.randn(B, K, N)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
