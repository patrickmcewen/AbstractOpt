"""PyTorch reference: Matmul followed by dropout and softmax."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, K, N, p=0.0):
        super().__init__()
        self.linear = nn.Linear(K, N, bias=False)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return torch.softmax(self.dropout(self.linear(x)), dim=-1)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims["K"], dims["N"], 0.0]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
