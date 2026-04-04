"""PyTorch reference: Gated Delta Net decode (single-step recurrent).

Linear attention with gated delta rule. Updates recurrent state and produces output.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, num_q_heads, num_v_heads, head_dim):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        # Learnable parameters
        self.A_log = nn.Parameter(torch.randn(num_v_heads) * 0.1)
        self.dt_bias = nn.Parameter(torch.zeros(num_v_heads))

    def forward(self, q, k, v, state, a, b):
        """
        q: (B, num_q_heads, D)
        k: (B, num_q_heads, D)  — num_k_heads == num_q_heads
        v: (B, num_v_heads, D)
        state: (B, num_v_heads, D, D) — recurrent state [H, V, K] layout
        a: (B, num_v_heads) — input-dependent decay
        b: (B, num_v_heads) — update gate input
        Returns: output (B, num_v_heads, D), new_state (B, num_v_heads, D, D)
        """
        B, HV, D = v.shape

        # Gate computation
        g = torch.exp(-torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float()))  # (B, HV)
        beta = torch.sigmoid(b.float())  # (B, HV)

        # Expand Q, K to match V heads
        ratio = HV // self.num_q_heads
        q_exp = q.float().repeat_interleave(ratio, dim=1)  # (B, HV, D)
        k_exp = k.float().repeat_interleave(ratio, dim=1)

        state_f = state.float()  # (B, HV, D, D) — [V, K] layout
        # state is [H, V, K] so state[b, h] is (V, K) = (D, D)
        # For delta rule: state[b,h] @ k gives (V,) = value retrieval
        # We need state in [K, V] form for k @ state = (1,K) @ (K,V) = (1,V)

        new_state = torch.zeros_like(state_f)
        output = torch.zeros(B, HV, D, dtype=torch.float32, device=q.device)

        for bi in range(B):
            for h in range(HV):
                s = state_f[bi, h].transpose(-1, -2)  # (D,D) -> (K,V) from (V,K)
                q_h = q_exp[bi, h].unsqueeze(0)  # (1, K)
                k_h = k_exp[bi, h].unsqueeze(0)  # (1, K)
                v_h = v[bi, h].float().unsqueeze(0)  # (1, V)
                g_h = g[bi, h]
                beta_h = beta[bi, h]

                old_s = g_h * s  # decay
                old_v = k_h @ old_s  # (1, V) retrieved value
                new_v = beta_h * v_h + (1.0 - beta_h) * old_v  # blended
                s_remove = k_h.T @ (k_h @ old_s)  # (K, V)
                s_update = k_h.T @ new_v  # (K, V)
                s_new = old_s - s_remove + s_update

                o = self.scale * (q_h @ s_new)  # (1, V)
                output[bi, h] = o.squeeze(0)
                new_state[bi, h] = s_new.transpose(-1, -2)  # back to (V, K)

        return output, new_state


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    HQ = dims["num_q_heads"]
    HV = dims["num_v_heads"]
    D = dims["head_dim"]
    q = torch.randn(B, HQ, D)
    k = torch.randn(B, HQ, D)
    v = torch.randn(B, HV, D)
    state = torch.randn(B, HV, D, D) * 0.01
    a = torch.randn(B, HV)
    b = torch.randn(B, HV)
    return [q, k, v, state, a, b]


def get_init_inputs(dims):
    return [dims["num_q_heads"], dims["num_v_heads"], dims["head_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
