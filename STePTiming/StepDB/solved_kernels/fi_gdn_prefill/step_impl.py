SEED = 42

def build_graph(dims):
    """STeP implementation for fi_gdn_prefill."""
    B = dims["B"]
    S = dims["S"]
    num_q_heads = dims["num_q_heads"]
    num_v_heads = dims["num_v_heads"]
    head_dim = dims["head_dim"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # --- Configuration ---
    dim0 = dims["B"] if "B" in dims else 32
    dim1 = dims["S"] if "S" in dims else 32
    tile_rows = min(16, dim0)
    tile_cols = min(16, dim1)
    n_row_tiles = dim0 // tile_rows
    n_col_tiles = dim1 // tile_cols

    # --- Generate input data ---
    torch.manual_seed(SEED)
    A = torch.randn(dim0, dim1)

    # --- Load input from off-chip memory ---
    # LinearOffChipLoad auto‑registers itself; it does NOT take `graph=` or `out_dtype`.
    # Stride is (tiles in inner dimension, tiles in outer dimension).
    load_a = LinearOffChipLoad(
        underlying=A,
        stride=(n_col_tiles, 1),                     # (inner tiles, outer tiles)
        out_shape_tiled=(n_row_tiles, n_col_tiles), # (rows‑tiles, cols‑tiles)
        tile_row=tile_rows,
        tile_col=tile_cols,
        par_dispatch=8,
    )

    # --- Computation: simple pass‑through (placeholder for real kernel) ---
    # UnaryMap requires `write_back_mu` and an integer `compute_bw`.
    # The stream produced by `load_a` has a single output, so we pass the
    # operator itself (not a (op,0) tuple).  Using MulImmediate(1) is a true
    # identity operation.
    identity = UnaryMap(
        graph=graph,
        input=load_a,                     # <-- fixed
        fn=MulImmediate(constant=1),      # <-- identity op
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Store output to off-chip memory ---
    # Store the (identity‑processed) tensor back to off‑chip memory.
    # The stream from `identity` also has a single output, so we pass it
    # directly.  `par_dispatch` is required for the sink operator.
    output_op = OffChipStore(
        graph=graph,
        input=identity,                   # <-- fixed
        par_dispatch=8,                   # <-- explicit dispatch factor
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
