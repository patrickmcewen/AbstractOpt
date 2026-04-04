SEED = 42

def build_graph(dims):
    """STeP implementation for fi_gdn_decode."""
    B = dims["B"]
    num_q_heads = dims["num_q_heads"]
    num_v_heads = dims["num_v_heads"]
    head_dim = dims["head_dim"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # --- Configuration ---
    dim0 = dims["B"] if "B" in dims else 32
    dim1 = dims["num_q_heads"] if "num_q_heads" in dims else 32
    tile_rows = min(16, dim0)
    tile_cols = min(16, dim1)
    n_row_tiles = dim0 // tile_rows
    n_col_tiles = dim1 // tile_cols

    # --- Generate input data ---
    torch.manual_seed(SEED)
    A = torch.randn(dim0, dim1)

    # --- Load input from off-chip memory ---
    # Load the placeholder tensor as a tiled stream.
    # LinearOffChipLoad auto‑registers itself; it does NOT take `graph=` or
    # any `out_dtype` / `compute_bw` arguments.
    load_a = LinearOffChipLoad(
        underlying=A,
        stride=(n_col_tiles, 1),                     # (inner tiles, outer tiles)
        out_shape_tiled=(n_row_tiles, n_col_tiles), # (rows‑tiles, cols‑tiles)
        tile_row=tile_rows,
        tile_col=tile_cols,
        par_dispatch=8,
    )

    # --- Computation: identity pass-through (replace with actual kernel logic) ---
    # Simple identity operation using UnaryMap.
    # UnaryMap requires `write_back_mu` and an integer `compute_bw`.
    identity = UnaryMap(
        graph=graph,
        input=load_a,                     # single‑output stream, no tuple needed
        fn=AddImmediate(constant=0),      # adds zero → identity
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Store output to off-chip memory ---
    # Store the (identity‑processed) tensor back to off‑chip memory.
    output_op = OffChipStore(
        graph=graph,
        input=identity,                   # single‑output stream
        par_dispatch=8,                   # required dispatch factor
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
