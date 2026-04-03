def build_graph(dims):
    """STeP implementation for sum_reduction."""
    M = dims["M"]
    K = dims["K"]
    reduce_dim = dims["reduce_dim"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # TODO: Load inputs with LinearOffChipLoad
    # TODO: Implement computation with STeP operators
    # TODO: Store output with OffChipStore

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
