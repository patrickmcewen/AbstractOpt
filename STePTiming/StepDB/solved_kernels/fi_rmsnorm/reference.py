"""PyTorch reference: RMS Layer Normalization (FlashInfer-style).

Computes x * rsqrt(mean(x^2) + eps) * weight. Used in Llama/DeepSeek transformers.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, K, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(K))

    def forward(self, x):
        x_f32 = x.float()
        inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * inv_rms * self.weight.float()).to(x.dtype)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims["K"], dims.get("eps", 1e-6)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
