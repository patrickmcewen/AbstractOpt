def build_graph(dims):
    """STeP implementation for matmul_scale_residual."""
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # Load weight tensor first (matches the reference model's weight generation)
    # No seed is set here, so we use the default RNG state exactly as the reference.
    linear_weight = torch.nn.Linear(K, N, bias=False)
    W = linear_weight.weight.t().contiguous()   # shape [K, N]

    # Load input activation tensor (deterministic seed as in the reference)
    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)

    # Tile sizes (default to 16 if not provided)
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)

    # Ensure dimensions are divisible by tile sizes
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    # ----------------------------------------------------------------------
    #  Load activation tiles
    # ----------------------------------------------------------------------
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # ----------------------------------------------------------------------
    #  Load weight tiles
    # ----------------------------------------------------------------------
    w_load = LinearOffChipLoad(
        underlying=W,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=4,
    )

    # ----------------------------------------------------------------------
    #  Perform tiled matrix multiplication (accumulate over K dimension)
    # ----------------------------------------------------------------------
    # Increase compute bandwidth to reduce compute cycles per tile.
    # Must remain an int.
    compute_bw = 8192
    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,                     # accumulate over the K‑tiles
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    #  Broadcast matmul result for reuse (needed for residual addition)
    # ----------------------------------------------------------------------
    broadcast = Broadcast(
        graph=graph,
        input=matmul,
        num_consumers=2,
    )
    # ----------------------------------------------------------------------
    #  Compute out * 0.5 using one copy of the broadcasted result
    # ----------------------------------------------------------------------
    half = UnaryMap(
        graph=graph,
        input=(broadcast, 1),   # second copy for scaling
        fn=MulImmediate(constant=0.5),
        # No need to write back to the memory unit for this intermediate result.
        write_back_mu=False,
        compute_bw=compute_bw,
    )
    # ----------------------------------------------------------------------
    #  Add the original result (first copy) and the scaled half (out + out*0.5)
    # ----------------------------------------------------------------------
    scaled = BinaryMap(
        graph=graph,
        in1=(broadcast, 0),     # original matmul output
        in2=half,
        fn=Add(),
        # Final result will be stored; intermediate write‑back is unnecessary.
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    # ----------------------------------------------------------------------
    #  Store the final output back to off‑chip memory
    # ----------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=scaled,
        par_dispatch=4,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
