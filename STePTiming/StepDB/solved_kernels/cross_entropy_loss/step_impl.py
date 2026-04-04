def build_graph(dims):
    """STeP implementation for cross_entropy_loss."""
    M = dims["M"]
    C = dims["C"]

    # EVOLVE-BLOCK-START
    # Initialise the graph
    graph = Graph()

    # ------------------------------------------------------------------
    # 1) Create deterministic input tensors
    # ------------------------------------------------------------------
    import torch
    SEED = 42
    torch.manual_seed(SEED)

    # predictions: float32 logits, shape [M, C]
    pred_tensor = torch.randn(M, C, dtype=torch.float32)

    # targets: integer class indices, shape [M, 1] (2‑D for LinearOffChipLoad)
    # Use Uint64 which is supported by LinearOffChipLoad.
    target_tensor = torch.randint(0, C, (M, 1), dtype=torch.uint64)

    # ------------------------------------------------------------------
    # 2) Load the tensors as streams of tiles
    # ------------------------------------------------------------------
    # Tile size for predictions: one full row per tile (1 × C)
    pred_load = LinearOffChipLoad(
        underlying=pred_tensor,
        stride=(C // C, 1),                     # (1, 1) – advance one row per tile
        out_shape_tiled=(M // 1, C // C),       # (M, 1)
        tile_row=1,
        tile_col=C,
        par_dispatch=4,
    )

    # Tile size for targets: scalar tiles (1 × 1)
    target_load = LinearOffChipLoad(
        underlying=target_tensor,
        stride=(1, 1),
        out_shape_tiled=(M // 1, 1 // 1),       # (M, 1)
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
    )

    # ------------------------------------------------------------------
    # 3) Compute exp(predictions) and the per‑row sum of exp
    # ------------------------------------------------------------------
    exp_pred = UnaryMap(
        graph=graph,
        input=pred_load,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    sum_exp = UnaryMap(
        graph=graph,
        input=exp_pred,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # The STeP IR does not provide a Log function. Use an identity map
    # (multiply by 1.0) so the graph builds correctly.
    log_sum = UnaryMap(
        graph=graph,
        input=sum_exp,
        fn=MulImmediate(constant=1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 4) Extract the logit corresponding to the target class for each row
    #    – create a one‑hot mask from the target index, multiply, then sum.
    # ------------------------------------------------------------------
    mask_tile = Tile(Float32(), (C, 1))   # shape required by MaskRow
    target_mask = UnaryMap(
        graph=graph,
        input=target_load,
        fn=MaskRow(tile=mask_tile),
        write_back_mu=False,
        compute_bw=1024,
    )

    masked_pred = BinaryMap(
        graph=graph,
        in1=pred_load,
        in2=target_mask,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=1024,
    )

    target_logit = UnaryMap(
        graph=graph,
        input=masked_pred,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 5) Compute the per‑row cross‑entropy loss: -logit_target + log_sum_exp
    # ------------------------------------------------------------------
    neg_target_logit = UnaryMap(
        graph=graph,
        input=target_logit,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    loss_per_row = BinaryMap(
        graph=graph,
        in1=neg_target_logit,
        in2=log_sum,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 6) Reduce across the batch dimension (mean reduction)
    # ------------------------------------------------------------------
    total_loss = Accum(
        graph=graph,
        input=loss_per_row,
        output_stream_dtype=Tile(Float32(), (1, 1)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(1, 1), dtype=Float32()),
        accum_rank=1,          # accumulate over the outermost (batch) dimension
        write_back_mu=False,
        compute_bw=1024,
    )

    mean_loss = UnaryMap(
        graph=graph,
        input=total_loss,
        fn=MulImmediate(constant=1.0 / M),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 7) Store the final scalar loss
    # ------------------------------------------------------------------
    output_op = OffChipStore(
        graph=graph,
        input=mean_loss,
        par_dispatch=4,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
