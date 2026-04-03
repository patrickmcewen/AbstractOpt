"""PyTorch reference: Top-k sampling from probability distribution.

Filters to top-k highest probability tokens, renormalizes, then samples.
Note: compute_gold returns the filtered probability distribution (not samples)
since sampling is non-deterministic.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, top_k):
        # top_k is per-row: (M,) int tensor
        M, V = probs.shape
        result = torch.zeros_like(probs)
        for i in range(M):
            k = int(top_k[i].item())
            topk_vals, topk_idx = torch.topk(probs[i], k)
            topk_probs = topk_vals / topk_vals.sum()
            result[i, topk_idx] = topk_probs
        return result


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, V = dims["M"], dims["V"]
    logits = torch.randn(M, V)
    probs = torch.softmax(logits, dim=-1)
    k = dims.get("top_k", 50)
    top_k = torch.full((M,), k, dtype=torch.int32)
    return [probs, top_k]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
