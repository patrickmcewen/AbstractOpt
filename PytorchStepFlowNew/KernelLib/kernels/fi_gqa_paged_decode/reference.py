"""PyTorch reference: Grouped Query Attention decode.

Single query token per batch attending to KV cache. GQA ratio = num_q_heads / num_kv_heads.
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
        q: (B, num_q_heads, head_dim)
        k: (B, S, num_kv_heads, head_dim)  — KV cache
        v: (B, S, num_kv_heads, head_dim)
        Returns: (B, num_q_heads, head_dim)
        """
        B, S, _, D = k.shape
        gqa_ratio = self.num_q_heads // self.num_kv_heads

        # Expand KV heads to match Q heads
        k_exp = k.repeat_interleave(gqa_ratio, dim=2)  # (B, S, num_q_heads, D)
        v_exp = v.repeat_interleave(gqa_ratio, dim=2)

        q_f = q.float().unsqueeze(2)  # (B, num_q_heads, 1, D)
        k_f = k_exp.float().permute(0, 2, 1, 3)  # (B, num_q_heads, S, D)
        v_f = v_exp.float().permute(0, 2, 1, 3)  # (B, num_q_heads, S, D)

        attn = (q_f @ k_f.transpose(-1, -2)) * self.scale  # (B, H, 1, S)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v_f).squeeze(2)  # (B, H, D)
        return out


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    S = dims["S"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    D = dims["head_dim"]
    q = torch.randn(B, num_q_heads, D)
    k = torch.randn(B, S, num_kv_heads, D)
    v = torch.randn(B, S, num_kv_heads, D)
    return [q, k, v]


def get_init_inputs(dims):
    return [dims["num_q_heads"], dims["num_kv_heads"], dims["head_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
