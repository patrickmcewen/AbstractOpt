"""PyTorch reference: GEMM (Matrix Multiply) — tiled M,N in the STeP variant."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, W):
        return torch.matmul(A, W)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    A = torch.randn(M, K)
    model = torch.nn.Linear(K, N, bias=False)
    W = model.weight.T.detach().clone().contiguous()  # [K, N]
    return A, W


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    A, W = get_inputs(dims)
    return torch.matmul(A, W)
