"""PyTorch reference: Routed MoE with top-k expert selection (no torch.where).

Full MoE layer with routing: each token selects top-k experts,
computes expert(x) = down(silu(gate(x)) * up(x)) for each selected expert,
and sums the weighted expert outputs.

  output[i] = sum_j( weight[i,j] * expert_j(x[i]) )  for j in top-k experts

Reimplemented to iterate over tokens and their selected experts directly,
avoiding torch.where.
"""
import torch
import torch.nn.functional as F

SEED = 42


def compute_gold(dims):
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    torch.manual_seed(SEED)

    # Must match step_impl RNG order exactly:
    # gate_weights[0..n], up_weights[0..n], down_weights[0..n], x, router_w
    gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    up_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]
    x = torch.randn(B, D)
    router_w = torch.randn(D, n_experts)

    # Router: top-k selection
    router_logits = x @ router_w  # [B, n_experts]
    _, expert_indices = torch.topk(router_logits, n_active, dim=-1)
    expert_weights_raw, _ = torch.topk(router_logits, n_active, dim=-1)
    expert_weights = torch.softmax(expert_weights_raw, dim=-1)  # [B, n_active]

    # Compute MoE output: iterate over tokens and their selected experts
    y = torch.zeros(B, D)
    with torch.no_grad():
        for token_idx in range(B):
            for k in range(n_active):
                expert_id = expert_indices[token_idx, k].item()
                xi = x[token_idx:token_idx + 1]  # [1, D]
                gate_out = xi @ gate_weights[expert_id]
                up_out = xi @ up_weights[expert_id]
                projected = F.silu(gate_out) * up_out
                down_out = projected @ down_weights[expert_id]  # [1, D]
                y[token_idx] += (expert_weights[token_idx, k] * down_out).squeeze(0)

    return y
