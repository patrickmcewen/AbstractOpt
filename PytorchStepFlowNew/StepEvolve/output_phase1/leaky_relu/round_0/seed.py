def build_graph(dims):
    """STeP implementation for leaky_relu."""
    M = dims["M"]
    K = dims["K"]
    negative_slope = dims["negative_slope"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # TODO: Load inputs with LinearOffChipLoad
    # TODO: Implement computation with STeP operators
    # TODO: Store output with OffChipStore

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
