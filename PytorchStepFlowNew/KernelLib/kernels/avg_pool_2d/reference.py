"""PyTorch reference: 2D average pooling."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, C, H, W = dims["B"], dims["C"], dims["H"], dims["W"]
    return [torch.randn(B, C, H, W)]


def get_init_inputs(dims):
    return [dims["kernel_size"], dims["stride"], dims["padding"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
