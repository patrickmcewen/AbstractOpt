# StepBench/baselines/layernorm.py
"""LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

Input shape: (B, F, D1, D2). Normalization is over the last 3 dims (F, D1, D2).

Since F*D1*D2 is contiguous in memory for each batch element, we can reshape
x to (B, F*D1*D2) without permutation.  Full-row tiles (tile_col = F*D1*D2)
let RowWiseSum compute the complete per-row reduction in one shot.

nn.LayerNorm has learnable weight and bias of shape (F, D1, D2), flattened
to (F*D1*D2,) for the element-wise multiply and add.
"""
import torch
import torch.nn as nn
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, init_fn
from step_py.ops import OffChipLoad, UnaryMap, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42
NR_ITERS = 4


def build_graph(dims):
    B = dims["batch_size"]
    F = dims["features"]
    D1 = dims["dim1"]
    D2 = dims["dim2"]

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    ln = nn.LayerNorm((F, D1, D2))
    weight = ln.weight.detach().reshape(-1)  # (N,) — ones by default
    bias = ln.bias.detach().reshape(-1)      # (N,) — zeros by default

    torch.manual_seed(SEED)
    x = torch.randn(B, F, D1, D2, dtype=torch.float32)

    x_load = OffChipLoad(underlying=x, stride=(1, 0),
                         out_shape_tiled=(1, 1),
                         tile_row=B, tile_col=F*D1*D2,
                         par_dispatch=4)

    sum_x = 


    graph = infer_broadcast(graph)
    return graph, _