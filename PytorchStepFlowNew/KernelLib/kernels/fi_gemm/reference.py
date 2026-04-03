"""PyTorch reference: General Matrix Multiplication C = A @ B^T (FlashInfer-style).

Standard GEMM with transposed B, as used in linear layers of LLMs.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B.T)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, N, K = dims["M"], dims["N"], dims["K"]
    return [torch.randn(M, K), torch.randn(N, K)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
