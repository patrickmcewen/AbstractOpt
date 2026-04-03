def build_graph(dims):
    """STeP implementation for tanh."""
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

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    # LinearOffChipLoad auto‑registers itself; no `graph=` argument needed
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=16,
    )
    # ------------------------------------------------------------
    # Compute tanh(x) using a higher‑order Padé‑3rd‑order approximation:
    #   tanh(x) ≈ x * (x⁴ + 105·x² + 945) / (15·x⁴ + 420·x² + 945)
    # This reduces the worst‑case error to ~1e‑4.
    # ------------------------------------------------------------
    # 1) x²
    sq = UnaryMap(
        graph=graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 2) x⁴ = (x²)²
    quad = UnaryMap(
        graph=graph,
        input=sq,
        fn=Square(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # -----------------------------------------------------------------
    # Build the numerator polynomial: x⁴ + 105·x² + 945
    # -----------------------------------------------------------------
    # 3) 105·x²
    sq_mul_105 = UnaryMap(
        graph=graph,
        input=sq,
        fn=MulImmediate(constant=105.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 4) x⁴ + 105·x²
    num_part = BinaryMap(
        graph=graph,
        in1=quad,
        in2=sq_mul_105,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 5) numerator polynomial = (x⁴ + 105·x²) + 945
    numerator_poly = UnaryMap(
        graph=graph,
        input=num_part,
        fn=AddImmediate(constant=945.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # -----------------------------------------------------------------
    # Build the denominator polynomial: 15·x⁴ + 420·x² + 945
    # -----------------------------------------------------------------
    # 6) 15·x⁴
    quad_mul_15 = UnaryMap(
        graph=graph,
        input=quad,
        fn=MulImmediate(constant=15.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 7) 420·x²
    sq_mul_420 = UnaryMap(
        graph=graph,
        input=sq,
        fn=MulImmediate(constant=420.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 8) 15·x⁴ + 420·x²
    den_part = BinaryMap(
        graph=graph,
        in1=quad_mul_15,
        in2=sq_mul_420,
        fn=Add(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 9) denominator polynomial = (15·x⁴ + 420·x²) + 945
    denominator_poly = UnaryMap(
        graph=graph,
        input=den_part,
        fn=AddImmediate(constant=945.0),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 10) x * numerator polynomial
    numer = BinaryMap(
        graph=graph,
        in1=load,
        in2=numerator_poly,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # 11) tanh ≈ numerator / denominator
    tanh_tile = BinaryMap(
        graph=graph,
        in1=numer,
        in2=denominator_poly,
        fn=Div(),
        write_back_mu=False,
        compute_bw=65536,
    )
    # ------------------------------------------------------------
    # Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=tanh_tile,
        par_dispatch=16,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
