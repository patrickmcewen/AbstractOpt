"""STeP implementation: GEMM with TileMN tiling via the Linear kernel."""

SEED = 42


def build_graph(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m = dims.get("tile_m", 16)
    tile_n = dims.get("tile_n", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)
    model = torch.nn.Linear(K, N, bias=False)
    W = model.weight.T.detach().clone().contiguous()  # [K, N]

    step_graph = Graph()

    linear = Linear(
        step_graph=step_graph,
        input=A,
        weight=W,
        tile_config=LinearTileConfig(m=tile_m, k=K, n=tile_n),
        comp_bw=1024,
        write_back_mu=True,
        par_dispatch=4,
    )

    output = OffChipStore(
        graph=step_graph,
        input=linear,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
