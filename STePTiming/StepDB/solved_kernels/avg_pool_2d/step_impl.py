SEED = 42

def build_graph(dims):
    """STeP implementation for avg_pool_2d."""
    B = dims["B"]
    C = dims["C"]
    H = dims["H"]
    W = dims["W"]
    kernel_size = dims["kernel_size"]
    stride = dims["stride"]
    padding = dims["padding"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # --- Configuration ---
    dim0 = dims["B"] if "B" in dims else 32
    dim1 = dims["C"] if "C" in dims else 32
    tile_rows = min(16, dim0)
    tile_cols = min(16, dim1)
    n_row_tiles = dim0 // tile_rows
    n_col_tiles = dim1 // tile_cols

    # --- Generate input data ---
    torch.manual_seed(SEED)
    A = torch.randn(dim0, dim1)

    # --- Load input from off-chip memory ---
    # Load the input tensor as a stream of tiles.
    # LinearOffChipLoad is a SOURCE operator and does NOT take a `graph`
    # argument.  The correct signature is:
    #   LinearOffChipLoad(underlying, stride, out_shape_tiled,
    #                     tile_row, tile_col, par_dispatch, ...)
    load_a = LinearOffChipLoad(
        underlying=A,
        stride=(n_col_tiles, 1),
        out_shape_tiled=(n_row_tiles, n_col_tiles),
        tile_row=tile_rows,
        tile_col=tile_cols,
        par_dispatch=4,   # modest parallelism; can be increased later
    )

    # --- Compute a global average per (B, C) channel ---
    # 1) Reshape the 4‑D input (B, C, H, W) into a 2‑D matrix (B*C, H*W)
    #    This reshaping is done once on the host side (allowed for data
    #    generation) before the tensor is streamed into the accelerator.
    torch.manual_seed(SEED)
    A = torch.randn(dim0, dim1, H, W)
    A2 = A.view(dim0 * dim1, H * W)   # shape: (B*C, H*W)

    # 2) Load the reshaped matrix as a stream of tiles.
    #    Use a single column tile that spans the whole spatial dimension
    #    so that RowWiseSum will produce the full sum per row.
    tile_rows = 1                                   # one row per tile
    tile_cols = H * W                               # full spatial dimension
    n_row_tiles = (dim0 * dim1) // tile_rows
    n_col_tiles = 1                                 # only one column tile
    load_a = LinearOffChipLoad(
        underlying=A2,
        stride=(n_col_tiles, 1),
        out_shape_tiled=(n_row_tiles, n_col_tiles),
        tile_row=tile_rows,
        tile_col=tile_cols,
        par_dispatch=4,
    )

    # 3) Row‑wise sum reduces the spatial dimension → shape (B*C, 1)
    row_sum = UnaryMap(
        graph=graph,
        input=load_a,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # 4) Scale by 1/(H*W) to obtain the mean.
    mean = UnaryMap(
        graph=graph,
        input=row_sum,
        fn=MulImmediate(constant=1.0 / (H * W)),
        write_back_mu=False,
        compute_bw=1024,
    )

    # 5) Store the per‑channel means back to off‑chip memory.
    output_op = OffChipStore(
        graph=graph,
        input=mean,
        par_dispatch=4,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
