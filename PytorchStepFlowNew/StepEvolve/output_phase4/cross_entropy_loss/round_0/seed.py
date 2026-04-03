def build_graph(dims):
    """STeP implementation for cross_entropy_loss."""
    M = dims["M"]
    C = dims["C"]

    # EVOLVE-BLOCK-START
    graph = Graph()

    # TODO: Load inputs with LinearOffChipLoad
    # TODO: Implement computation with STeP operators
    # TODO: Store output with OffChipStore

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
