def build_graph(dims):
    """STeP implementation for batched_matmul."""
    B = dims["B"]
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # EVOLVE-BLOCK-START
    # ----------------------------------------------------------------------
    # 1️⃣  Create the graph
    # ----------------------------------------------------------------------
    graph = Graph()

    # ----------------------------------------------------------------------
    # 2️⃣  Tiling parameters (default to full size if not supplied)
    # ----------------------------------------------------------------------
    tile_b = dims.get("tile_b", B)          # batch tiling
    tile_m = dims.get("tile_m", M)          # rows of A / output
    tile_k = dims.get("tile_k", K)          # reduction dim
    tile_n = dims.get("tile_n", N)          # cols of B / output

    # All tiled dimensions must divide the original sizes
    assert B % tile_b == 0, f"B={B} not divisible by tile_b={tile_b}"
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    # ----------------------------------------------------------------------
    # 3️⃣  Create concrete input tensors (deterministic seed)
    # ----------------------------------------------------------------------
    torch.manual_seed(SEED)
    A = torch.randn(B, M, K, dtype=torch.float32)
    Bmat = torch.randn(B, K, N, dtype=torch.float32)

    # ----------------------------------------------------------------------
    # 4️⃣  Load the two operands from off‑chip memory
    # ----------------------------------------------------------------------
    # A is tiled as [tile_m, tile_k] and broadcast across the N‑tiles.
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 0, 1),                     # (batch, M, N, K) strides
        out_shape_tiled=(
            1,                    # single batch tile
            M // tile_m,          # M‑tiles
            N // tile_n,          # N‑tiles (broadcast dimension)
            K // tile_k,          # K‑tiles (reduction)
        ),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=8,
    )

    # B is tiled as [tile_k, tile_n] and broadcast across the M‑tiles.
    # First load the weight tiles (no batch dimension).
    weight_load = LinearOffChipLoad(
        underlying=Bmat,
        stride=(0, 0, 1, K // tile_k),                     # (batch, M, N, K) strides
        out_shape_tiled=(
            M // tile_m,          # M‑tiles are not needed for weight; kept for shape compatibility
            N // tile_n,
            K // tile_k,
        ),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=8,
    )
    # Repeat the weight stream once per batch tile so that its outermost
    # dimension matches the activation stream's batch dimension.
    b_load = RepeatStatic(
        graph=graph,
        input=weight_load,
        repeat_factor=1,   # only one batch tile
    )

    # ----------------------------------------------------------------------
    # 5️⃣  Perform the tiled matrix multiplication
    # ----------------------------------------------------------------------
    compute_bw = 8192                     # increased compute bandwidth for faster compute
    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(),        # weight‑stationary matmul
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,                           # accumulate over the K‑tiles
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    # 6️⃣  Store the result back to off‑chip memory
    # ----------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=8,
        store_file_name="output",         # mandatory name
    )

    # ----------------------------------------------------------------------
    # 7️⃣  Finalise the graph
    # ----------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
