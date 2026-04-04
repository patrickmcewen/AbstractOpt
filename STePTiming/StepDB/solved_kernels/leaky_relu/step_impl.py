def build_graph(dims):
    """STeP implementation for leaky_relu."""
    M = dims["M"]
    K = dims["K"]
    negative_slope = dims["negative_slope"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # ------------------------------------------------------------
    # Load the input tensor from off‑chip memory
    # ------------------------------------------------------------
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
        par_dispatch=16,                        # increased parallel HBM reads
    )
    # ------------------------------------------------------------
    # Compute Leaky ReLU: y = ((1+α)/2) * x + ((1‑α)/2) * |x|
    #   where α = negative_slope
    # ------------------------------------------------------------
    α = negative_slope
    coeff_pos = (1.0 + α) / 2.0          # multiplier for the positive part
    coeff_neg = (1.0 - α) / 2.0          # multiplier for the |x| part

    # 1) |x| = sqrt(x²)  -> Square -> Rsqrt -> Mul
    sq = UnaryMap(
        graph=graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )
    rs = UnaryMap(
        graph=graph,
        input=sq,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )
    abs_val = BinaryMap(
        graph=graph,
        in1=sq,
        in2=rs,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )

    # 2) coeff_pos * x
    pos_term = UnaryMap(
        graph=graph,
        input=load,
        fn=MulImmediate(constant=coeff_pos),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )

    # 3) coeff_neg * |x|
    neg_term = UnaryMap(
        graph=graph,
        input=abs_val,
        fn=MulImmediate(constant=coeff_neg),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )

    # 4) sum the two terms → final Leaky ReLU result
    leaky_relu = BinaryMap(
        graph=graph,
        in1=pos_term,
        in2=neg_term,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,                       # higher compute bandwidth
    )
    # ------------------------------------------------------------
    # Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=leaky_relu,
        par_dispatch=16,                        # increased parallel HBM writes
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
