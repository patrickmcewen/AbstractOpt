"""Benchmark: current functional emulator vs PCL-lite on a 4096x4096x4096 GEMM."""

import time
import sys
from pathlib import Path

import torch

# ── Current emulator setup ──────────────────────────────────────────────────
STEP_TL_SRC = str(Path(__file__).resolve().parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent / "step_tl" / "src" / "proto")
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

# ── PCL-lite setup ──────────────────────────────────────────────────────────
PCL_LITE = str(Path(__file__).resolve().parent.parent / "AccelOptStep" / "PCL-lite")
sys.path.insert(0, PCL_LITE)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
M, K, N = 4096, 4096, 4096
TILE = 16
SEED = 42

torch.manual_seed(SEED)
A = torch.randn(M, K)
B = torch.randn(K, N)
gold = A @ B


# ─────────────────────────────────────────────────────────────────────────────
# Current emulator
# ─────────────────────────────────────────────────────────────────────────────
def bench_current():
    from graph.graph import MultiDiGraph as Graph
    from rewrite.broadcast import infer_broadcast
    from step_py.ops import LinearOffChipLoad, BinaryMapAccum, OffChipStore, StepOps
    from step_py.functions import map_accum_fn, init_fn
    from step_py.datatype import Float32
    from step_py.functional import execute

    StepOps._counter = 0
    graph = Graph()

    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // TILE, 0, 1),
        out_shape_tiled=(M // TILE, N // TILE, K // TILE),
        tile_row=TILE, tile_col=TILE,
        par_dispatch=4,
    )
    b_load = LinearOffChipLoad(
        underlying=B,
        stride=(0, 1, N // TILE),
        out_shape_tiled=(M // TILE, N // TILE, K // TILE),
        tile_row=TILE, tile_col=TILE,
        par_dispatch=4,
    )
    matmul = BinaryMapAccum(
        graph=graph, in1=a_load, in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(TILE, TILE), dtype=Float32()),
        rank=1, write_back_mu=True, compute_bw=4096,
    )
    output_op = OffChipStore(
        graph=graph, input=matmul,
        par_dispatch=4, store_file_name="output",
    )
    graph = infer_broadcast(graph)

    # Warm up
    StepOps._counter = 0
    execute(graph, output_op)

    t0 = time.perf_counter()
    result = execute(graph, output_op)
    t1 = time.perf_counter()
    return result, t1 - t0


# ─────────────────────────────────────────────────────────────────────────────
# PCL-lite emulator
# ─────────────────────────────────────────────────────────────────────────────
def bench_pcl_lite():
    import step
    from sympy import Symbol

    Msym, Ksym, Nsym = Symbol("M"), Symbol("K"), Symbol("N")
    ctx = {Msym: M // TILE, Ksym: K // TILE, Nsym: N // TILE}

    # Tile the inputs: (M//T, K//T, T, T) and (K//T, N//T, T, T)
    # PCL-lite shape is reversed: [innermost, ..., outermost]
    # Stream shape (outer dims) = [Ksym, Nsym, Msym] means tensor is (M//T, N//T, K//T)
    # but PCL-lite reverses, so shape = [Ksym, Nsym, Msym]

    # Tile A into (M//T, K//T, T, T) and B into (K//T, N//T, T, T)
    a_tiled = A.reshape(M // TILE, TILE, K // TILE, TILE).permute(0, 2, 1, 3)
    b_tiled = B.reshape(K // TILE, TILE, N // TILE, TILE).permute(0, 2, 1, 3)

    # Broadcast: A needs N//T dim, B needs M//T dim
    # For Zip, both must have same shape. Expand to (M//T, N//T, K//T, T, T)
    a_exp = a_tiled.unsqueeze(1).expand(M // TILE, N // TILE, K // TILE, TILE, TILE)
    b_exp = b_tiled.unsqueeze(0).expand(M // TILE, N // TILE, K // TILE, TILE, TILE)

    # PCL-lite stores shape reversed: [Ksym, Nsym, Msym]
    e0 = step.Stream("E0", step.Tile("float", [TILE, TILE]), 3, [Ksym, Nsym, Msym])
    e0.ctx = ctx
    e0.data = [a_exp]  # shape: (M//T, N//T, K//T, T, T)

    e1 = step.Stream("E1", step.Tile("float", [TILE, TILE]), 3, [Ksym, Nsym, Msym])
    e1.ctx = ctx
    e1.data = [b_exp]

    class TileMatmul(step.Fn):
        def __init__(self):
            super().__init__(
                "TileMatmul",
                step.STuple((step.Tile("float", [TILE, TILE]), step.Tile("float", [TILE, TILE]))),
                step.Tile("float", [TILE, TILE]),
            )
        def apply(self, input):
            return [input[0] @ input[1]]

    class TileAdd(step.Fn):
        def __init__(self):
            super().__init__(
                "TileAdd",
                step.Tile("float", [TILE, TILE]),
                step.Tile("float", [TILE, TILE]),
            )
        def getInit(self):
            return [torch.zeros(TILE, TILE)]
        def apply(self, state, input):
            return [state[0] + input[0]]

    fn_matmul = TileMatmul()
    fn_add = TileAdd()

    # Warm up (small)
    # Skip warmup for PCL-lite since it's purely Python loops

    t0 = time.perf_counter()

    e2 = step.Zip().apply((e0, e1))
    e3 = step.Map(fn=fn_matmul).apply(e2)
    e4 = step.Accum(fn=fn_add, b=1).apply(e3)

    t1 = time.perf_counter()

    # Result is in e4.data[0], shape (M//T, N//T, T, T)
    result = e4.data[0].permute(0, 2, 1, 3).reshape(M, N)
    return result, t1 - t0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GEMM: {M}x{K}x{N}, tile={TILE}")
    print(f"Tile grid: {M//TILE}x{N//TILE}x{K//TILE} = {(M//TILE)*(N//TILE)*(K//TILE)} tile triples\n")

    print("Running current emulator...")
    res_cur, t_cur = bench_current()
    err_cur = (res_cur - gold).abs().max().item()
    print(f"  Time: {t_cur:.3f}s  max_err: {err_cur:.2e}")

    print("\nRunning PCL-lite emulator...")
    res_pcl, t_pcl = bench_pcl_lite()
    err_pcl = (res_pcl - gold).abs().max().item()
    print(f"  Time: {t_pcl:.3f}s  max_err: {err_pcl:.2e}")

    print(f"\nSpeedup: {t_pcl / t_cur:.1f}x")
