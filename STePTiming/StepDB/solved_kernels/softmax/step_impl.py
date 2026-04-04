def build_graph(dims):
    """STeP implementation for softmax."""
    M = dims["M"]
    K = dims["K"]
    graph = Graph()

    # EVOLVE-BLOCK-START
    # ----------------------------------------------------------------------
    # Build a tiled softmax: exp(x) / sum_exp(x) over the last dimension K
    # ----------------------------------------------------------------------
    import torch

    # Tiling parameters (must divide the dimensions)
    tile_m = dims.get("tile_m", M)          # rows per tile
    tile_k = dims.get("tile_k", K)          # cols per tile
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    # Load the input matrix A as a stream of [tile_m, tile_k] tiles
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                     # advance outer dim per tile row
        out_shape_tiled=(M // tile_m, K // tile_k),   # (outer_rows, outer_cols)
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # 1) exp(A)
    exp_op = UnaryMap(
        graph=graph,
        input=load,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # 2) row‑wise sum of exp(A) → shape [tile_m, 1]
    row_sum = UnaryMap(
        graph=graph,
        input=exp_op,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # 3) Use the row‑wise sum directly for division (no broadcast needed)

    # 4) Softmax: exp(A) / sum_exp(A) (per‑row)
    softmax = BinaryMap(
        graph=graph,
        in1=exp_op,
        in2=row_sum,          # use row-wise sum directly
        fn=Div(),
        write_back_mu=True,              # write result back to memory unit
        compute_bw=1024,
    )

    # 5) Store the result
    output_op = OffChipStore(
        graph=graph,
        input=softmax,
        par_dispatch=4,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
