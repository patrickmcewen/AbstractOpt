def build_graph(dims):
    """STeP implementation for group_norm."""
    B = dims["B"]
    C = dims["C"]
    H = dims["H"]
    W = dims["W"]
    G = dims["G"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # Deterministic input tensor
    torch.manual_seed(SEED)
    # Original shape (B, C, H, W)
    X_full = torch.randn(B, C, H, W)

    # Reshape for GroupNorm: combine batch and groups, flatten group channels and spatial dims
    C_per_group = C // G
    M = B * G
    K = C_per_group * H * W
    X = X_full.view(B, G, C_per_group, H, W).reshape(M, K)

    # Tile configuration
    # Use full‑row tiles by default to minimise hierarchical stop‑tokens
    # and reduce the total number of streamed groups.
    tile_m = dims.get("tile_m", M)
    tile_k = dims.get("tile_k", K)  # full column tile
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    eps = dims.get("eps", 1e-5)

    # Load input as tiled stream
    # Increase off‑chip request parallelism to saturate the memory bandwidth
    act = LinearOffChipLoad(
        underlying=X,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=256,
    )

    # ---------- Mean computation ----------
    row_sum = UnaryMap(
        graph=graph,
        input=act,
        fn=RowWiseSum(),
        write_back_mu=False,
        # Double the compute bandwidth – reduces per‑tile compute latency
        compute_bw=8192,
    )
    # Compute -(row_sum / K) in a single step to avoid an extra op.
    neg_mean = UnaryMap(
        graph=graph,
        input=row_sum,
        fn=MulImmediate(constant=-1.0 / K),
        write_back_mu=False,
        compute_bw=8192,
    )
    diff = BinaryMap(
        graph=graph,
        in1=act,
        in2=neg_mean,
        fn=Add(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # ---------- Variance computation ----------
    diff_sq = UnaryMap(
        graph=graph,
        input=diff,
        fn=Square(),
        write_back_mu=False,
        compute_bw=8192,
    )
    var_sum = UnaryMap(
        graph=graph,
        input=diff_sq,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=8192,
    )
    var = UnaryMap(
        graph=graph,
        input=var_sum,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=8192,
    )
    var_eps = UnaryMap(
        graph=graph,
        input=var,
        fn=AddImmediate(constant=eps),
        write_back_mu=False,
        compute_bw=8192,
    )
    inv_std = UnaryMap(
        graph=graph,
        input=var_eps,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # ---------- Normalization ----------
    norm = BinaryMap(
        graph=graph,
        in1=diff,
        in2=inv_std,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # Store result
    # Boost off‑chip store parallelism as well
    output_op = OffChipStore(
        graph=graph,
        input=norm,
        par_dispatch=256,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
