"""PyTorch reference: FP8 block-scale MoE (DeepSeek V3/R1 style).

Includes no-aux routing (sigmoid + group topk) and two grouped-GEMMs with SwiGLU.
Simplified to float32 (no actual FP8 quantization) for gold computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    """Simplified DeepSeek V3 MoE block.

    For gold computation, operates in float32 without actual FP8 quantization.
    The routing logic (sigmoid + group-topk + global-topk) is preserved exactly.
    """

    def __init__(self, H, I, E, n_active=8, n_group=8, topk_group=4, scaling_factor=1.0):
        super().__init__()
        self.H = H
        self.I = I
        self.E = E
        self.n_active = n_active
        self.n_group = n_group
        self.topk_group = topk_group
        self.scaling_factor = scaling_factor
        # Expert weights: gate_up (2*I, H) and down (H, I) per expert
        self.gate_up = nn.Parameter(torch.randn(E, 2 * I, H) * 0.01)
        self.down = nn.Parameter(torch.randn(E, H, I) * 0.01)
        self.routing = nn.Parameter(torch.randn(E))  # routing bias

    def forward(self, x, routing_logits):
        """
        Args:
            x: (T, H) hidden states
            routing_logits: (T, E) raw routing logits
        Returns:
            (T, H) output
        """
        T = x.shape[0]
        E = self.E
        device = x.device

        # --- Routing ---
        s = torch.sigmoid(routing_logits.float())
        s_biased = s + self.routing.float()

        group_size = E // self.n_group
        s_grouped = s_biased.view(T, self.n_group, group_size)
        top2_vals, _ = torch.topk(s_grouped, k=2, dim=2)
        group_scores = top2_vals.sum(dim=2)  # (T, n_group)

        _, group_idx = torch.topk(group_scores, k=self.topk_group, dim=1)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(2).expand(T, self.n_group, group_size).reshape(T, E)

        neg_inf = torch.finfo(torch.float32).min
        scores_pruned = s_biased.masked_fill(score_mask == 0, neg_inf)
        _, topk_idx = torch.topk(scores_pruned, k=self.n_active, dim=1)

        # Weights from unbiased scores
        M = torch.zeros_like(s)
        M.scatter_(1, topk_idx, 1.0)
        weights = s * M
        weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * self.scaling_factor

        # --- Expert computation ---
        output = torch.zeros(T, self.H, dtype=torch.float32, device=device)
        for e in range(E):
            sel = (topk_idx == e).any(dim=1)
            if not sel.any():
                continue
            idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
            x_e = x[idx].float()  # (Tk, H)
            g1 = x_e @ self.gate_up[e].float().T  # (Tk, 2I)
            gate, up = g1[:, :self.I], g1[:, self.I:]
            hidden = F.silu(up) * gate  # SwiGLU
            out_e = hidden @ self.down[e].float().T  # (Tk, H)
            w = weights[idx, e].unsqueeze(1)
            output.index_add_(0, idx, out_e * w)

        return output


def get_inputs(dims):
    torch.manual_seed(SEED)
    T, H, E = dims["T"], dims["H"], dims["E"]
    return [torch.randn(T, H), torch.randn(T, E)]


def get_init_inputs(dims):
    return [
        dims["H"], dims["I"], dims["E"],
        dims.get("n_active", 8),
        dims.get("n_group", 8),
        dims.get("topk_group", 4),
        dims.get("scaling_factor", 1.0),
    ]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
