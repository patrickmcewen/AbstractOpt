SEED = 42

def build_graph(dims):
    """STeP implementation for fi_moe_fp8_block_scale."""
    T = dims["T"]
    H = dims["H"]
    I = dims["I"]
    E = dims["E"]
    n_active = dims["n_active"]
    n_group = dims["n_group"]
    topk_group = dims["topk_group"]
    scaling_factor = dims["scaling_factor"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # --- Configuration ---
    dim0 = dims["T"] if "T" in dims else 32
    dim1 = dims["H"] if "H" in dims else 32
    tile_rows = min(16, dim0)
    tile_cols = min(16, dim1)
    n_row_tiles = dim0 // tile_rows
    n_col_tiles = dim1 // tile_cols

    # --- Generate input data ---
    torch.manual_seed(SEED)
    A = torch.randn(dim0, dim1)

    # --- Load input activations [T, H] as tiles ---
    torch.manual_seed(SEED)
    x = torch.randn(dim0, dim1)

    load_x = LinearOffChipLoad(
        underlying=x,
        stride=(dim1 // tile_cols, 1),               # (tiles per row, outer)
        out_shape_tiled=(dim0 // tile_rows, dim1 // tile_cols),
        tile_row=tile_rows,
        tile_col=tile_cols,
        par_dispatch=4,
    )

    # --- Load weight matrix [H, H] (triggered by activation stream) ---
    W = torch.randn(dim1, dim1)

    # Load weight matrix [H, H] (triggered by activation stream)
    # For matmul X @ W we need weight tiles of shape (K, N) = (tile_cols, tile_cols)
    weight_load = LinearOffChipLoadRef(
        graph=graph,
        ref=load_x,
        underlying=W,
        stride=(dim1 // tile_cols, 1),
        out_shape_tiled=(dim1 // tile_cols, dim1 // tile_cols),
        tile_row=tile_cols,   # <-- rows must match the shared K dimension
        tile_col=tile_cols,
        par_dispatch=4,
    )

    # --- Tiled matrix multiplication: X @ W ---
    matmul_out = BinaryMapAccum(
        graph=graph,
        in1=load_x,
        in2=weight_load,
        fn=map_accum_fn.Matmul(weight_transposed=False),
        init_fn=init_fn.Zero(shape=(tile_rows, tile_cols), dtype=Float32()),
        rank=1,                     # accumulate over inner dimension
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Store result to off‑chip memory ---
    output_op = OffChipStore(
        graph=graph,
        input=matmul_out,
        par_dispatch=4,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
