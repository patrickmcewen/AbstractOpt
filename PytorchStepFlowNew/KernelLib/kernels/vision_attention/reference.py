"""PyTorch reference: Vision self-attention: reshape spatial to sequence, multihead attention, residual + layernorm."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, C, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(C, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)         # (B, H*W, C)
        attn_out, _ = self.attn(seq, seq, seq)
        out = self.ln(seq + attn_out)
        return out.transpose(1, 2).reshape(B, C, H, W)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, C, H, W = dims["B"], dims["C"], dims["H"], dims["W"]
    return [torch.randn(B, C, H, W)]


def get_init_inputs(dims):
    return [dims["C"], dims["num_heads"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
