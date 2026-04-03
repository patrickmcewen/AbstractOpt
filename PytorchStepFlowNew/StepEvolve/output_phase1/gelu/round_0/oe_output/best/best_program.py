def build_graph(dims):
    """STeP implementation for gelu."""
    M = dims["M"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # ------------------------------------------------------------
    # Configuration & input generation
    # ------------------------------------------------------------
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # Use a fixed seed for reproducibility (matches the reference)
    torch.manual_seed(42)
    A = torch.randn(M, K)

    # ------------------------------------------------------------
    # Load the input tensor from off‑chip memory
    # ------------------------------------------------------------
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                # (inner tiles, outer tiles)
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=16,                        # increased parallelism
    )

    # ------------------------------------------------------------
    # Accurate GELU using the tanh formulation:
    # gelu(x) ≈ x * sigmoid( 2 * sqrt(2/π) * (x + 0.044715 * x³) )
    # ------------------------------------------------------------
    # 1) x²
    x_sq = UnaryMap(
        graph=graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 2) x³ = x² * x
    x_cu = BinaryMap(
        graph=graph,
        in1=x_sq,
        in2=load,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 3) 0.044715 * x³
    term_cu = UnaryMap(
        graph=graph,
        input=x_cu,
        fn=MulImmediate(constant=0.044715),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 4) x + 0.044715 * x³
    inner = BinaryMap(
        graph=graph,
        in1=load,
        in2=term_cu,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 5) Multiply by constant 2 * sqrt(2/π) ≈ 1.5957691216
    arg = UnaryMap(
        graph=graph,
        input=inner,
        fn=MulImmediate(constant=1.5957691216),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 6) exp(arg)
    exp_arg = UnaryMap(
        graph=graph,
        input=arg,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=131072,   # higher bandwidth for heavy op
    )
    # 7) denom = exp(arg) + 1
    denom = UnaryMap(
        graph=graph,
        input=exp_arg,
        fn=AddImmediate(constant=1.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 8) sigmoid = exp(arg) / (exp(arg) + 1)
    sigmoid = BinaryMap(
        graph=graph,
        in1=exp_arg,
        in2=denom,
        fn=Div(),
        write_back_mu=False,
        compute_bw=131072,   # higher bandwidth for division
    )
    # 9) GELU ≈ x * sigmoid
    gelu = BinaryMap(
        graph=graph,
        in1=load,
        in2=sigmoid,
        fn=Mul(),
        write_back_mu=True,    # write‑back to memory unit to reduce store latency
        compute_bw=65536,
    )

    # ------------------------------------------------------------
    # Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=gelu,
        par_dispatch=16,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
