"""PyTorch reference: Multi-layer perceptron with ReLU activations."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, D_in, D_hidden, D_out, n_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(D_in, D_hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(D_hidden, D_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(D_hidden, D_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, D_in = dims["M"], dims["D_in"]
    return [torch.randn(M, D_in)]


def get_init_inputs(dims):
    return [dims["D_in"], dims["D_hidden"], dims["D_out"], dims.get("n_layers", 2)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
