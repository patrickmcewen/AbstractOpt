def build_graph(dims):
    """STeP implementation for sum_reduction."""
    M = dims["M"]
    K = dims["K"]
    reduce_dim = dims["reduce_dim"]

    # EVOLVE-BLOCK-START
    # ----------------------------------------------------------------------
    # Build a tiled sum‑reduction: torch.sum(x, dim=reduce_dim, keepdim=True)
    # Supports reduction over the last dimension (‑1 or 1) only.
    # ----------------------------------------------------------------------
    graph = Graph()

    # Only last‑dimension reduction is implemented for simplicity
    assert reduce_dim in (-1, 1), "Only reduction over the last dimension is supported"

    # Tile sizes – must divide the matrix dimensions
    tile_m = dims.get("tile_m", M)          # rows per tile
    # Use a full‑column tile so that the row‑wise sum already produces the
    # final reduced result. This eliminates the need for an extra Accum stage
    # and reduces the number of streamed tiles, lowering cycle count.
    tile_k = K          # cols per tile (no column tiling)
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    # With tile_k == K the following assert is always true, but we keep it for
    # clarity and future safety.
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # Create the input tensor (deterministic seed)
    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    # ------------------------------------------------------------------
    # 1) Load the input matrix A as a stream of [tile_m, tile_k] tiles
    # ------------------------------------------------------------------
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                     # advance outer dim per tile row
        out_shape_tiled=(M // tile_m, K // tile_k),   # (outer_rows, outer_cols)
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=32,  # further increased parallel dispatch for higher bandwidth
    )

    # ------------------------------------------------------------------
    # 2) Row‑wise sum reduces the K‑dimension inside each tile
    #    Output tiles have shape [tile_m, 1]
    # ------------------------------------------------------------------
    # Enable write‑back to the memory unit to reduce load/store overhead
    # The row‑wise sum already produces the reduced tile; we don’t need to
    # write it back to the memory unit because the next stage (Accum or Store)
    # will consume the stream directly.  Disabling write_back_mu removes the
    # extra store latency.  A modest compute bandwidth (e.g. 1024) is sufficient
    # for this lightweight reduction.
    row_sum = UnaryMap(
        graph=graph,
        input=load,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 3) If the K dimension is tiled, accumulate the partial sums across
    #    the tile‑wise reductions.
    # ------------------------------------------------------------------
    # Since we tile the full K dimension, the RowWiseSum already yields the
    # complete per‑row reduction. No further accumulation is required.
    result = row_sum

    # ------------------------------------------------------------------
    # 4) Store the final reduced tensor
    # ------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=result,
        par_dispatch=32,  # further increased parallel dispatch for faster off‑chip writes
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
