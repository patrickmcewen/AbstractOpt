def build_graph(dims):
    """STeP implementation for mse_loss."""
    M = dims["M"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    # ------------------------------------------------------------------
    # 1) Initialise the STeP graph.
    # ------------------------------------------------------------------
    graph = Graph()
    # ------------------------------------------------------------------
    # 2) Set tile size (1×1) so each element is a separate tile.
    # ------------------------------------------------------------------
    tile_m = 1
    tile_k = 1
    assert M % tile_m == 0 and K % tile_k == 0, "Tile size must divide dimensions"

    # ------------------------------------------------------------------
    # 2) Load the two input tensors.
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    pred_tensor = torch.randn(M, K)
    target_tensor = torch.randn(M, K)

    load_pred = LinearOffChipLoad(
        underlying=pred_tensor,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=32,
    )

    load_target = LinearOffChipLoad(
        underlying=target_tensor,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=32,
    )

    # ------------------------------------------------------------------
    # 3) (pred - target) = pred + (-1 * target)
    # ------------------------------------------------------------------
    neg_target = UnaryMap(
        graph=graph,
        input=load_target,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    diff = BinaryMap(
        graph=graph,
        in1=load_pred,
        in2=neg_target,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 4) Square the difference.
    # ------------------------------------------------------------------
    sq = UnaryMap(
        graph=graph,
        input=diff,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 5) Sum all squared elements → scalar tile.
    #    The stream shape after loading is (1, M, K); we reduce both
    #    inner dimensions (accum_rank=2) to obtain a single 1×1 tile.
    # ------------------------------------------------------------------
    total_sum = Accum(
        graph=graph,
        input=sq,
        output_stream_dtype=Tile(Float32(), (1, 1)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(1, 1), dtype=Float32()),
        accum_rank=2,
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 6) Divide by the number of elements to get the mean.
    # ------------------------------------------------------------------
    mean = UnaryMap(
        graph=graph,
        input=total_sum,
        fn=MulImmediate(constant=1.0 / (M * K)),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 7) Store the final MSE value.
    # ------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=mean,
        par_dispatch=32,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
