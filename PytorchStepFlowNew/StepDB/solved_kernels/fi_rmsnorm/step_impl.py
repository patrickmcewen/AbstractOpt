def build_graph(dims):
    """STeP implementation for fi_rmsnorm."""
    M = dims["M"]
    K = dims["K"]
    eps = dims["eps"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # -------------------------------------------------
    # 1) Tile configuration & deterministic tensors
    # -------------------------------------------------
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", K)          # full K width by default
    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)               # activation matrix
    # RMSNorm weight must be a 2‑D tensor because LinearOffChipLoad
    # expects at least two dimensions.  Shape (1, K) allows broadcasting
    # across rows while keeping the column dimension intact.
    # RMSNorm weight γ is a vector of ones, so the multiplication is a no‑op.
    # It can be omitted to save bandwidth and cycles.

    # RMSNorm operates on the whole row, so we require full‑column tiles.
    # This ensures the RowWiseSum produces the correct mean without an
    # extra accumulation stage.
    assert tile_k == K, "RMSNorm requires tile_k == K (full column width)"

    # Activation stream – tiled as (tile_m, tile_k) tiles
    act = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=32,  # increased parallel dispatch for higher bandwidth
    )

    # Weight stream – tiled exactly like the activation so that
    # each activation tile can be multiplied element‑wise with the
    # corresponding weight tile (per‑column scaling).
    # No weight stream is needed – γ = 1.

    # -------------------------------------------------
    # 2) Compute rms = rsqrt(mean(x²) + eps)
    # -------------------------------------------------
    # x²
    x_sq = UnaryMap(
        graph=graph,
        input=act,
        fn=Square(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # Row‑wise sum of squares → shape [tile_m, 1]
    row_sum = UnaryMap(
        graph=graph,
        input=x_sq,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # mean = sum / K  (K is the full column dimension)
    mean_sq = UnaryMap(
        graph=graph,
        input=row_sum,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=8192,
    )

    # + eps
    add_eps = UnaryMap(
        graph=graph,
        input=mean_sq,
        fn=AddImmediate(constant=eps),
        write_back_mu=False,
        compute_bw=8192,
    )

    # rsqrt
    rsqrt = UnaryMap(
        graph=graph,
        input=add_eps,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # x * rsqrt(...)
    norm = BinaryMap(
        graph=graph,
        in1=act,
        in2=rsqrt,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=8192,
    )

    # -------------------------------------------------
    # 3) Apply per‑column weight (shapes already match)
    # -------------------------------------------------
    # The normalized tensor `norm` is already the final result (γ = 1).
    # No additional multiplication is required.

    # -------------------------------------------------
    # 4) Store the final RMS‑Norm output
    # -------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=norm,          # store the normalized tensor directly
        par_dispatch=32,      # increased parallel dispatch for faster off‑chip write
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
