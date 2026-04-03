"""PyTorch reference: Gated Delta Net prefill (sequential over sequence).

Processes full sequence, updating recurrent state at each step.
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
        self.A_log = nn.Parameter(torch.randn(num_v_heads) * 0.1)
        self.dt_bias = nn.Parameter(torch.zeros(num_v_heads))

    def forward(self, q, k, v, state, a, b):
        """
        q: (B, S, num_q_heads, D)
        k: (B, S, num_q_heads, D)
        v: (B, S, num_v_heads, D)
        state: (B, num_v_heads, D, D) — initial recurrent state
        a: (B, S, num_v_heads) — per-step decay
        b: (B, S, num_v_heads) — per-step gate
        Returns: output (B, S, num_v_heads, D), final_state (B, num_v_heads, D, D)
        """
        B, S, HQ, D = q.shape
        HV = self.num_v_heads
        ratio = HV // HQ
        device = q.device

        q_exp = q.float().repeat_interleave(ratio, dim=2)  # (B, S, HV, D)
        k_exp = k.float().repeat_interleave(ratio, dim=2)

        output = torch.zeros(B, S, HV, D, device=device)
        final_state = state.float().clone()  # (B, HV, D, D) in (V, K) layout

        for bi in range(B):
            for h in range(HV):
                s = final_state[bi, h].transpose(-1, -2)  # (K, V) from (V, K)
                for t in range(S):
                    q_t = q_exp[bi, t, h].unsqueeze(0)
                    k_t = k_exp[bi, t, h].unsqueeze(0)
                    v_t = v[bi, t, h].float().unsqueeze(0)
                    x = a[bi, t, h].float() + self.dt_bias[h].float()
                    g = torch.exp(-torch.exp(self.A_log[h].float()) * F.softplus(x))
                    beta = torch.sigmoid(b[bi, t, h].float())

                    old_s = g * s
                    old_v = k_t @ old_s
                    new_v = beta * v_t + (1.0 - beta) * old_v
                    s = old_s - k_t.T @ (k_t @ old_s) + k_t.T @ new_v

                    output[bi, t, h] = (self.scale * (q_t @ s)).squeeze(0)
                final_state[bi, h] = s.transpose(-1, -2)

        return output, final_state


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    S = dims["S"]
    HQ = dims["num_q_heads"]
    HV = dims["num_v_heads"]
    D = dims["head_dim"]
    q = torch.randn(B, S, HQ, D)
    k = torch.randn(B, S, HQ, D)
    v = torch.randn(B, S, HV, D)
    state = torch.randn(B, HV, D, D) * 0.01
    a = torch.randn(B, S, HV)
    b = torch.randn(B, S, HV)
    return [q, k, v, state, a, b]


def get_init_inputs(dims):
    return [dims["num_q_heads"], dims["num_v_heads"], dims["head_dim"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
