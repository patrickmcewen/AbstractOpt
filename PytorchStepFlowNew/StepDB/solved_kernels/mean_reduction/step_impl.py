def build_graph(dims):
    """STeP implementation for mean_reduction."""
    M = dims["M"]
    K = dims["K"]
    reduce_dim = dims["reduce_dim"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # ------------------------------------------------------------------
    # 1) Validate reduction dimension and tile sizes
    # ------------------------------------------------------------------
    # Only reduction over the last dimension (‑1 or 1) is supported.
    assert reduce_dim in (-1, 1), "Only reduction over the last dimension is supported"

    # Tile the rows; use a full‑column tile so that RowWiseSum yields the final mean.
    tile_m = dims.get("tile_m", M)
    tile_k = K                     # full K dimension per tile

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    # tile_k == K, so the following assert is always true, but we keep it for clarity.
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # ------------------------------------------------------------------
    # 2) Create a deterministic input tensor
    # ------------------------------------------------------------------
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K)

    # ------------------------------------------------------------------
    # 3) Load the input matrix A as a stream of [tile_m, tile_k] tiles
    # ------------------------------------------------------------------
    # Increase parallel HBM dispatch to improve bandwidth and lower cycle count
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                     # advance outer dim per tile row
        out_shape_tiled=(M // tile_m, K // tile_k),   # (outer_rows, outer_cols)
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=32,   # higher parallelism than the default 4
    )

    # ------------------------------------------------------------------
    # 4) Row‑wise sum reduces the K dimension inside each tile → shape [tile_m, 1]
    # ------------------------------------------------------------------
    row_sum = UnaryMap(
        graph=graph,
        input=load,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,          # int, as required
    )

    # ------------------------------------------------------------------
    # 5) Scale by 1/K to obtain the mean (keepdim=True)
    # ------------------------------------------------------------------
    mean = UnaryMap(
        graph=graph,
        input=row_sum,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 6) Store the final mean tensor
    # ------------------------------------------------------------------
    # Use a higher dispatch count for off‑chip stores to match the load bandwidth
    output_op = OffChipStore(
        graph=graph,
        input=mean,
        par_dispatch=32,   # increased from 4 to reduce store latency
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
