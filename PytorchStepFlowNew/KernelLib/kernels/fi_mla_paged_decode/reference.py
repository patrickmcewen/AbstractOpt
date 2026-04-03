"""PyTorch reference: Multi-head Latent Attention (MLA) decode.

DeepSeek V3 attention: separate nope/pe query projections attend to compressed KV + KV PE.
"""
import math
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, num_heads, ckv_dim, kpe_dim):
        super().__init__()
        self.num_heads = num_heads
        self.ckv_dim = ckv_dim
        self.kpe_dim = kpe_dim
        self.scale = 1.0 / math.sqrt(ckv_dim + kpe_dim)

    def forward(self, q_nope, q_pe, ckv, kpe):
        """
        q_nope: (B, num_heads, ckv_dim)
        q_pe: (B, num_heads, kpe_dim)
        ckv: (B, S, ckv_dim) — compressed KV cache
        kpe: (B, S, kpe_dim) — KV positional encoding cache
        Returns: (B, num_heads, ckv_dim)
        """
        qn = q_nope.float()  # (B, H, Dckv)
        qp = q_pe.float()    # (B, H, Dkpe)
        kc = ckv.float()     # (B, S, Dckv)
        kp = kpe.float()     # (B, S, Dkpe)

        # Attention: nope part + pe part
        logits = torch.einsum('bhd,bsd->bhs', qn, kc) + torch.einsum('bhd,bsd->bhs', qp, kp)
        logits = logits * self.scale

        attn = torch.softmax(logits, dim=-1)  # (B, H, S)
        out = torch.einsum('bhs,bsd->bhd', attn, kc)  # (B, H, Dckv)
        return out


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    S = dims["S"]
    H = dims["num_heads"]
    Dckv = dims["ckv_dim"]
    Dkpe = dims["kpe_dim"]
    q_nope = torch.randn(B, H, Dckv)
    q_pe = torch.randn(B, H, Dkpe)
    ckv = torch.randn(B, S, Dckv)
    kpe = torch.randn(B, S, Dkpe)
    return [q_nope, q_pe, ckv, kpe]


def get_init_inputs(dims):
    return [dims["num_heads"], dims["ckv_dim"], dims["kpe_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
