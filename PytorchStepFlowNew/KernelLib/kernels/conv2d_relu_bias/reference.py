"""PyTorch reference: Conv2D followed by ReLU and bias addition."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, C_in, C_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, padding=kernel_size // 2)
        self.bias = nn.Parameter(torch.randn(C_out, 1, 1))

    def forward(self, x):
        return torch.relu(self.conv(x)) + self.bias


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, C_in, H, W = dims["B"], dims["C_in"], dims["H"], dims["W"]
    return [torch.randn(B, C_in, H, W)]


def get_init_inputs(dims):
    return [dims["C_in"], dims["C_out"], dims["kernel_size"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
