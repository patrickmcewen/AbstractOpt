"""PyTorch reference: Multi-head Latent Attention (MLA) causal prefill.

Full sequence attention with causal masking using compressed KV.
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
        q_nope: (B, S, num_heads, ckv_dim)
        q_pe: (B, S, num_heads, kpe_dim)
        ckv: (B, S, ckv_dim)
        kpe: (B, S, kpe_dim)
        Returns: (B, S, num_heads, ckv_dim)
        """
        B, S, H, Dckv = q_nope.shape

        qn = q_nope.float().permute(0, 2, 1, 3)  # (B, H, S, Dckv)
        qp = q_pe.float().permute(0, 2, 1, 3)     # (B, H, S, Dkpe)
        kc = ckv.float()                            # (B, S, Dckv)
        kp = kpe.float()                            # (B, S, Dkpe)

        logits = torch.einsum('bhqd,bkd->bhqk', qn, kc) + torch.einsum('bhqd,bkd->bhqk', qp, kp)
        logits = logits * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=logits.device), diagonal=1).bool()
        logits.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(logits, dim=-1)  # (B, H, S, S)
        out = torch.einsum('bhqk,bkd->bhqd', attn, kc)  # (B, H, S, Dckv)
        return out.permute(0, 2, 1, 3)  # (B, S, H, Dckv)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    S = dims["S"]
    H = dims["num_heads"]
    Dckv = dims["ckv_dim"]
    Dkpe = dims["kpe_dim"]
    q_nope = torch.randn(B, S, H, Dckv)
    q_pe = torch.randn(B, S, H, Dkpe)
    ckv = torch.randn(B, S, Dckv)
    kpe = torch.randn(B, S, Dkpe)
    return [q_nope, q_pe, ckv, kpe]


def get_init_inputs(dims):
    return [dims["num_heads"], dims["ckv_dim"], dims["kpe_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
