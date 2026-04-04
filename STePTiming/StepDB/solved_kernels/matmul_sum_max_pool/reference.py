"""PyTorch reference: Matmul followed by sum reduction, max, and average pooling."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.linear = nn.Linear(K, N)

    def forward(self, x):
        x = self.linear(x)                        # (M, N)
        x = x.sum(dim=-1, keepdim=True)            # (M, 1) sum reduction
        x = x.max(dim=0, keepdim=True)[0]          # (1, 1) max over batch
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
