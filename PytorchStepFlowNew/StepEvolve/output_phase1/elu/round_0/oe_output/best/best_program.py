def build_graph(dims):
    """STeP implementation for elu."""
    M = dims["M"]
    K = dims["K"]
    alpha = dims["alpha"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # Load input tensor from off‑chip memory
    torch.manual_seed(42)
    A = torch.randn(M, K)

    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # LinearOffChipLoad auto‑registers itself; no `graph=` argument needed
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),                # (inner tiles, outer tiles)
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=16,
    )

    # Compute absolute value |x|
    sq = UnaryMap(
        graph=graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=65536,
    )
    rs = UnaryMap(
        graph=graph,
        input=sq,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=65536,
    )
    abs_val = BinaryMap(
        graph=graph,
        in1=sq,
        in2=rs,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=65536,
    )

    # Positive part: relu(x) = (x + |x|) * 0.5
    sum_pos = BinaryMap(
        graph=graph,
        in1=load,
        in2=abs_val,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,
    )
    pos = UnaryMap(
        graph=graph,
        input=sum_pos,
        fn=MulImmediate(constant=0.5),
        write_back_mu=False,
        compute_bw=65536,
    )

    # exp(x)
    exp_tile = UnaryMap(
        graph=graph,
        input=load,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # exp(x) - 1
    exp_minus_one = UnaryMap(
        graph=graph,
        input=exp_tile,
        fn=AddImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # alpha * (exp(x) - 1)
    neg_branch = UnaryMap(
        graph=graph,
        input=exp_minus_one,
        fn=MulImmediate(constant=alpha),
        write_back_mu=False,
        compute_bw=65536,
    )

    # mask = (|x| - x) * 0.5 / |x|
    load_neg = UnaryMap(
        graph=graph,
        input=load,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    diff = BinaryMap(
        graph=graph,
        in1=abs_val,
        in2=load_neg,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,
    )
    half_diff = UnaryMap(
        graph=graph,
        input=diff,
        fn=MulImmediate(constant=0.5),
        write_back_mu=False,
        compute_bw=65536,
    )
    mask = BinaryMap(
        graph=graph,
        in1=half_diff,
        in2=abs_val,
        fn=Div(),
        write_back_mu=False,
        compute_bw=65536,
    )

    # masked negative contribution
    masked_neg = BinaryMap(
        graph=graph,
        in1=mask,
        in2=neg_branch,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=65536,
    )

    # final ELU output
    elu = BinaryMap(
        graph=graph,
        in1=pos,
        in2=masked_neg,
        fn=Add(),
        write_back_mu=True,
        compute_bw=65536,
    )

    # Store result back to off‑chip memory
    output_op = OffChipStore(
        graph=graph,
        input=elu,
        par_dispatch=16,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
