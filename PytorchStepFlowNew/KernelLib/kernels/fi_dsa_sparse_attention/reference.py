"""PyTorch reference: DeepSeek Sparse Attention (DSA).

MLA-style attention but only attending to top-k selected KV positions per query.
"""
import math
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, num_heads, ckv_dim, kpe_dim, topk):
        super().__init__()
        self.num_heads = num_heads
        self.ckv_dim = ckv_dim
        self.kpe_dim = kpe_dim
        self.topk = topk
        self.scale = 1.0 / math.sqrt(ckv_dim + kpe_dim)

    def forward(self, q_nope, q_pe, ckv, kpe, sparse_indices):
        """
        q_nope: (T, num_heads, ckv_dim)
        q_pe: (T, num_heads, kpe_dim)
        ckv: (S, ckv_dim) — full KV cache
        kpe: (S, kpe_dim)
        sparse_indices: (T, topk) — which KV positions each query attends to
        Returns: (T, num_heads, ckv_dim)
        """
        T, H, Dckv = q_nope.shape
        device = q_nope.device

        output = torch.zeros(T, H, Dckv, device=device)

        for t in range(T):
            idx = sparse_indices[t]  # (topk,)
            valid = idx >= 0
            idx_valid = idx[valid].long()

            if idx_valid.numel() == 0:
                continue

            kc = ckv[idx_valid].float()   # (n, Dckv)
            kp = kpe[idx_valid].float()   # (n, Dkpe)
            qn = q_nope[t].float()        # (H, Dckv)
            qp = q_pe[t].float()          # (H, Dkpe)

            logits = (qn @ kc.T + qp @ kp.T) * self.scale  # (H, n)
            attn = torch.softmax(logits, dim=-1)
            output[t] = (attn @ kc)

        return output


def get_inputs(dims):
    torch.manual_seed(SEED)
    T = dims["T"]
    S = dims["S"]
    H = dims["num_heads"]
    Dckv = dims["ckv_dim"]
    Dkpe = dims["kpe_dim"]
    topk = dims["topk"]
    q_nope = torch.randn(T, H, Dckv)
    q_pe = torch.randn(T, H, Dkpe)
    ckv = torch.randn(S, Dckv)
    kpe = torch.randn(S, Dkpe)
    # Random sparse indices (each query picks topk positions from [0, S))
    sparse_indices = torch.stack([torch.randperm(S)[:topk] for _ in range(T)])
    return [q_nope, q_pe, ckv, kpe, sparse_indices]


def get_init_inputs(dims):
    return [dims["num_heads"], dims["ckv_dim"], dims["kpe_dim"], dims["topk"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
