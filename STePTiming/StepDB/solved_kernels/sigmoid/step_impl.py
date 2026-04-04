def build_graph(dims):
    """STeP implementation for sigmoid."""
    M = dims["M"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # ------------------------------------------------------------
    # Load the input tensor from off‑chip memory
    # ------------------------------------------------------------
    torch.manual_seed(42)
    A = torch.randn(M, K)

    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    # LinearOffChipLoad auto‑registers itself; no `graph=` argument needed
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                # (inner tiles, outer tiles)
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=16,
    )
    # ------------------------------------------------------------
    # Compute sigmoid(x) = exp(x) / (1 + exp(x))
    # ------------------------------------------------------------
    # exp(x)
    exp_tile = UnaryMap(
        graph=graph,
        input=load,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=65536,
    )

    # 1 + exp(x)  (AddImmediate adds a scalar constant to each tile)
    denom_tile = UnaryMap(
        graph=graph,
        input=exp_tile,
        fn=AddImmediate(constant=1.0),
        write_back_mu=False,
        compute_bw=65536,
    )

    # sigmoid = exp(x) / (1 + exp(x))
    sigmoid_tile = BinaryMap(
        graph=graph,
        in1=exp_tile,
        in2=denom_tile,
        fn=Div(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # ------------------------------------------------------------
    # Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=sigmoid_tile,
        par_dispatch=16,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
