"""PyTorch reference: ResNet BasicBlock: two 3x3 convs with BN, ReLU, and shortcut."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C_out)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or C_in != C_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out),
            )
        self.eval()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, C_in, H, W = dims["B"], dims["C_in"], dims["H"], dims["W"]
    return [torch.randn(B, C_in, H, W)]


def get_init_inputs(dims):
    return [dims["C_in"], dims["C_out"], dims.get("stride", 1)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
