def build_graph(dims):
    """STeP implementation for matmul_dropout_softmax."""
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # ----------------------------------------------------------------------
    # Tiling parameters (default to 16 if not supplied)
    # ----------------------------------------------------------------------
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)
    par_dispatch = dims.get("par_dispatch", 4)
    compute_bw = dims.get("compute_bw", 4096)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    # ----------------------------------------------------------------------
    # Create deterministic tensors (seed must match reference)
    # ----------------------------------------------------------------------
    torch.manual_seed(SEED)
    X = torch.randn(M, K, dtype=torch.float32)
    W = torch.randn(K, N, dtype=torch.float32)

    # ----------------------------------------------------------------------
    # Load activation and weight tiles
    # ----------------------------------------------------------------------
    act_load = LinearOffChipLoad(
        underlying=X,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    weight_load = LinearOffChipLoad(
        underlying=W,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    # ----------------------------------------------------------------------
    # Linear layer: X @ W  (weight‑stationary GEMM)
    # ----------------------------------------------------------------------
    gemm = BinaryMapAccum(
        graph=graph,
        in1=act_load,
        in2=weight_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    # Dropout (training) – scale by (1‑p).  p defaults to 0.0 if not supplied.
    # ----------------------------------------------------------------------
    p = dims.get("p", 0.0)
    dropout_scaled = UnaryMap(
        graph=graph,
        input=gemm,
        fn=MulImmediate(constant=1.0 - p),
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    # Softmax over the last dimension N
    # ----------------------------------------------------------------------
    exp_op = UnaryMap(
        graph=graph,
        input=dropout_scaled,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    row_sum = UnaryMap(
        graph=graph,
        input=exp_op,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    softmax = BinaryMap(
        graph=graph,
        in1=exp_op,
        in2=row_sum,
        fn=Div(),
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    # Store the final softmax result
    # ----------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=softmax,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
