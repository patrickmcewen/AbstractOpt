"""PyTorch reference: Matmul + bias add followed by Swish, Tanh, GELU, and HardTanh activations."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.linear = nn.Linear(K, N)

    def forward(self, x):
        x = self.linear(x)
        x = x * torch.sigmoid(x)  # Swish
        x = torch.tanh(x)
        x = F.gelu(x)
        x = F.hardtanh(x)
        return x


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
