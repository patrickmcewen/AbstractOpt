def build_graph(dims):
    """STeP implementation for fi_gemm."""
    M = dims["M"]
    N = dims["N"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    # ------------------------------------------------------------
    # 1️⃣  Initialise the graph
    # ------------------------------------------------------------
    graph = Graph()

    # ------------------------------------------------------------
    # 2️⃣  Tiling parameters (default to 16 if not supplied)
    # ------------------------------------------------------------
    tile_m = dims.get("tile_m", 16)
    tile_n = dims.get("tile_n", 16)
    tile_k = dims.get("tile_k", 16)

    # ------------------------------------------------------------
    # 3️⃣  Validate that problem sizes are divisible by tile sizes
    # ------------------------------------------------------------
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # ------------------------------------------------------------
    # 4️⃣  Create deterministic input tensors (float32)
    # ------------------------------------------------------------
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(N, K, dtype=torch.float32)   # note: shape (N, K)

    # ------------------------------------------------------------
    # 5️⃣  Load the two operands from off‑chip memory
    # ------------------------------------------------------------
    # A is tiled as [tile_m, tile_k] and broadcast across N‑tiles.
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),                     # (M‑tiles, N‑tiles, K‑tiles) strides
        out_shape_tiled=(
            M // tile_m,          # M‑tiles
            N // tile_n,          # N‑tiles (broadcast dimension)
            K // tile_k,          # K‑tiles (reduction)
        ),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # B is tiled as [tile_k, tile_n] (the weight matrix) and broadcast across M‑tiles.
    b_load = LinearOffChipLoad(
        underlying=B,
        stride=(0, 1, N // tile_n),                     # (M‑tiles, N‑tiles, K‑tiles) strides
        out_shape_tiled=(
            M // tile_m,
            N // tile_n,
            K // tile_k,
        ),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=4,
    )

    # ------------------------------------------------------------
    # 6️⃣  Perform tiled matrix multiplication (A @ Bᵀ)
    # ------------------------------------------------------------
    compute_bw = 4096                         # integer compute bandwidth
    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(weight_transposed=True),   # B is used transposed
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,                                            # accumulate over K‑tiles
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ------------------------------------------------------------
    # 7️⃣  Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=4,
        store_file_name="output",          # mandatory name
    )

    # ------------------------------------------------------------
    # 8️⃣  Finalise the graph
    # ------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
