def build_graph(dims):
    """STeP implementation for fi_gemm."""
    M = dims["M"]
    N = dims["N"]
    K = dims["K"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # TODO: Load inputs with LinearOffChipLoad
    # TODO: Implement computation with STeP operators
    # TODO: Store output with OffChipStore

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
