"""Build initial programs (seeds) for OpenEvolve with EVOLVE-BLOCK markers."""

import re

from src.experience_store import ExperienceStore


def build_seed(store: ExperienceStore, kernel_name: str, dims: dict) -> str:
    """Build an initial program for OpenEvolve.

    Strategy:
    1. If the store has an exact match for kernel_name, wrap its build_graph body.
    2. Otherwise, generate a minimal scaffold.
    """
    best = store.get_best(kernel_name)
    if best is not None:
        return _wrap_existing(best.code)
    return _scaffold(kernel_name, dims)


def _wrap_existing(code: str) -> str:
    """Wrap an existing build_graph function body with EVOLVE-BLOCK markers."""
    match = re.search(r"(def build_graph\(dims\):.*?\n)(.*)", code, re.DOTALL)
    assert match, "Code does not contain def build_graph(dims):"

    signature_line = match.group(1)
    body = match.group(2)

    # Determine indentation from first non-empty body line
    indent = "    "
    for line in body.split("\n"):
        if line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            break

    return f"{signature_line}{indent}# EVOLVE-BLOCK-START\n{body}\n{indent}# EVOLVE-BLOCK-END\n"


def _scaffold(kernel_name: str, dims: dict) -> str:
    """Generate a minimal build_graph scaffold."""
    dim_unpacking = "\n".join(
        f'    {k} = dims["{k}"]' for k in dims
    )
    if not dim_unpacking:
        dim_unpacking = "    pass  # no dims specified"

    return f'''\
def build_graph(dims):
    """STeP implementation for {kernel_name}."""
{dim_unpacking}

    # EVOLVE-BLOCK-START
    graph = Graph()

    # TODO: Load inputs with LinearOffChipLoad
    # TODO: Implement computation with STeP operators
    # TODO: Store output with OffChipStore

    graph = infer_broadcast(graph)
    return graph, output_op
    # EVOLVE-BLOCK-END
'''
