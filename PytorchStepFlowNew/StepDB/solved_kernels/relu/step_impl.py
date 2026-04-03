def build_graph(dims):
    """STeP implementation for relu."""
    M = dims["M"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    # Initialize the computation graph
    graph = Graph()

    # ------------------------------------------------------------
    # Configuration & input generation
    # ------------------------------------------------------------
    M, K = dims["M"], dims["K"]
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
        stride=(K // tile_k, 1),                # stride in tiles (inner dim, outer dim)
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=8,                         # increased parallel HBM requests
    )

    # ------------------------------------------------------------
    # Apply ReLU – since ReLU isn’t available in the IR, we approximate
    # it with SiLU (Swish) which is a monotonic activation and works for
    # the purpose of this benchmark.
    # ------------------------------------------------------------
    # -----------------------------------------------------------------
    # Compute ReLU exactly: relu(x) = (x + |x|) / 2
    #   |x| = sqrt(x^2) = x^2 * Rsqrt(x^2)
    # Steps:
    #   1) sq   = x^2               (Square)
    #   2) rs   = 1 / sqrt(x^2)    (Rsqrt)
    #   3) abs  = sq * rs           (BinaryMap Mul)
    #   4) sum  = x + abs           (BinaryMap Add)
    #   5) relu = sum * 0.5         (UnaryMap MulImmediate)
    # -----------------------------------------------------------------
    sq = UnaryMap(
        graph=graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=16384,                       # higher compute bandwidth
    )

    rs = UnaryMap(
        graph=graph,
        input=sq,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=16384,                       # higher compute bandwidth
    )

    abs_val = BinaryMap(
        graph=graph,
        in1=sq,
        in2=rs,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=16384,                       # higher compute bandwidth
    )

    sum_val = BinaryMap(
        graph=graph,
        in1=load,
        in2=abs_val,
        fn=Add(),
        write_back_mu=False,
        compute_bw=16384,                       # higher compute bandwidth
    )

    relu = UnaryMap(
        graph=graph,
        input=sum_val,
        fn=MulImmediate(constant=0.5),
        write_back_mu=False,
        compute_bw=16384,                       # higher compute bandwidth
    )

    # ------------------------------------------------------------
    # Store the result back to off‑chip memory
    # ------------------------------------------------------------
    output = OffChipStore(
        graph=graph,
        input=relu,
        par_dispatch=8,                         # increased parallel HBM writes
        store_file_name="output",
    )

    # Final broadcast inference and return
    graph = infer_broadcast(graph)
    return graph, output
    # EVOLVE-BLOCK-END
