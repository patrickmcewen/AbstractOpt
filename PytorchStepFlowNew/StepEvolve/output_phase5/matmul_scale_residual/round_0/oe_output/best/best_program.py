def build_graph(dims):
    """STeP implementation for matmul_scale_residual."""
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # Determine tiling parameters (must divide the problem dimensions)
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)
    par_dispatch = dims.get("par_dispatch", 4)
    compute_bw = dims.get("compute_bw", 4096)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    # Seed for reproducibility
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float32)   # input activation
    W = torch.randn(K, N, dtype=torch.float32)   # linear weight

    # ------------------------------------------------------------------
    # 1) Load activations and weights as tiled streams
    # ------------------------------------------------------------------
    # Activations: (M//tile_m, N//tile_n, K//tile_k) tiles of shape [tile_m, tile_k]
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # Weights: (M//tile_m, N//tile_n, K//tile_k) tiles of shape [tile_k, tile_n]
    w_load = LinearOffChipLoad(
        underlying=W,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    # ------------------------------------------------------------------
    # 2) Tiled matrix multiplication (A @ W) with accumulation over K tiles
    # ------------------------------------------------------------------
    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,                     # accumulate over innermost K dimension
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ------------------------------------------------------------------
    # 3) Residual connection: out = matmul + 0.5 * matmul
    # ------------------------------------------------------------------
    # Broadcast the matmul result so we can use it twice
    matmul_bc = Broadcast(
        graph=graph,
        input=matmul,
        num_consumers=2,
    )

    # Scale one copy by 0.5
    scaled = UnaryMap(
        graph=graph,
        input=(matmul_bc, 1),
        fn=MulImmediate(constant=0.5),
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    # Add the original and the scaled copy
    residual = BinaryMap(
        graph=graph,
        in1=(matmul_bc, 0),
        in2=scaled,
        fn=Add(),
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ------------------------------------------------------------------
    # 4) Store the final tensor back to off‑chip memory
    # ------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=residual,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
