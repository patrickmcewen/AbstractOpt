def build_graph(dims):
    """STeP implementation for matmul_sum_max_pool."""
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # EVOLVE-BLOCK-START
    # --------------------------------------------------------------
    # 1) Initialise the graph and deterministic input tensors
    # --------------------------------------------------------------
    graph = Graph()

    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    # --------------------------------------------------------------
    # 2) Tiling parameters (must divide the dimensions)
    # --------------------------------------------------------------
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    # --------------------------------------------------------------
    # 3) Load the two operands of the linear layer
    # --------------------------------------------------------------
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),                 # advance M per tile row
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    b_load = LinearOffChipLoad(
        underlying=B,
        stride=(0, 1, N // tile_n),                 # advance N per tile column
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=4,
    )

    # --------------------------------------------------------------
    # 4) Tiled matrix‑multiply (weight‑stationary GEMM)
    # --------------------------------------------------------------
    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,                     # accumulate over the K‑dimension
        write_back_mu=True,
        compute_bw=4096,
    )

    # --------------------------------------------------------------
    # 5) Row‑wise sum → reduces the last dimension (N) keeping dim
    # --------------------------------------------------------------
    row_sum = UnaryMap(
        graph=graph,
        input=matmul,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )
    # `row_sum` now has shape (M, 1) tiled as (M//tile_m, 1)

    # --------------------------------------------------------------
    # 6) Batch‑wise sum → reduces the M dimension, producing a scalar tile
    # --------------------------------------------------------------
    total_sum = Accum(
        graph=graph,
        input=row_sum,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_m, 1), dtype=Float32()),
        accum_rank=1,                 # accumulate over the inner (M) dimension
        write_back_mu=False,
        compute_bw=1024,
    )

    # --------------------------------------------------------------
    # 7) Compute average (sum * 1/M) – implements the average‑pooling step
    # --------------------------------------------------------------
    avg = UnaryMap(
        graph=graph,
        input=total_sum,
        fn=MulImmediate(constant=1.0 / M),
        write_back_mu=False,
        compute_bw=1024,
    )

    # --------------------------------------------------------------
    # 8) Store the final scalar result
    # --------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=avg,
        par_dispatch=4,
        store_file_name="output",
    )

    # --------------------------------------------------------------
    # 8) Finalise the graph
    # --------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
