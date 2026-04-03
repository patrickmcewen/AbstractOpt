"""PyTorch reference: Grouped Query Attention causal prefill.

Full sequence Q attending to full KV with causal masking.
"""
import math
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v):
        """
        q: (B, S, num_q_heads, head_dim)
        k: (B, S, num_kv_heads, head_dim)
        v: (B, S, num_kv_heads, head_dim)
        Returns: (B, S, num_q_heads, head_dim)
        """
        B, S, _, D = q.shape
        gqa_ratio = self.num_q_heads // self.num_kv_heads

        k_exp = k.repeat_interleave(gqa_ratio, dim=2)
        v_exp = v.repeat_interleave(gqa_ratio, dim=2)

        q_f = q.float().permute(0, 2, 1, 3)  # (B, H, S, D)
        k_f = k_exp.float().permute(0, 2, 1, 3)
        v_f = v_exp.float().permute(0, 2, 1, 3)

        attn = (q_f @ k_f.transpose(-1, -2)) * self.scale  # (B, H, S, S)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
        attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v_f  # (B, H, S, D)
        return out.permute(0, 2, 1, 3)  # (B, S, H, D)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    S = dims["S"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    D = dims["head_dim"]
    q = torch.randn(B, S, num_q_heads, D)
    k = torch.randn(B, S, num_kv_heads, D)
    v = torch.randn(B, S, num_kv_heads, D)
    return [q, k, v]


def get_init_inputs(dims):
    return [dims["num_q_heads"], dims["num_kv_heads"], dims["head_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
