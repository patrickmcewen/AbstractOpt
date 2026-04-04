"""PyTorch reference: Cross-entropy loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return F.cross_entropy(predictions, targets)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, C = dims["M"], dims["C"]
    return [torch.randn(M, C), torch.randint(0, C, (M,))]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
