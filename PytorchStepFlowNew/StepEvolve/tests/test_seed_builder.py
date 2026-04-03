import unittest

from src.experience_store import Experience, ExperienceStore
from src.seed_builder import build_seed


SAMPLE_STEP_CODE = '''\
def build_graph(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    graph = Graph()
    a_load = LinearOffChipLoad(underlying=torch.randn(M, K), stride=(1, 0), out_shape_tiled=(1, 1), tile_row=M, tile_col=K)
    output = OffChipStore(graph=graph, input=a_load, par_dispatch=4, store_file_name="output")
    graph = infer_broadcast(graph)
    return graph, output
'''


class TestBuildSeed(unittest.TestCase):
    def test_direct_match(self):
        store = ExperienceStore()
        store.add(Experience("gemm", SAMPLE_STEP_CODE, 100.0, 1e-6, {}, []))

        seed = build_seed(store, "gemm", {"M": 32, "K": 48, "N": 64})
        assert "EVOLVE-BLOCK-START" in seed
        assert "EVOLVE-BLOCK-END" in seed
        assert "build_graph" in seed
        assert "LinearOffChipLoad" in seed

    def test_scaffold_when_no_match(self):
        store = ExperienceStore()

        seed = build_seed(store, "unknown_kernel", {"M": 32, "K": 48})
        assert "EVOLVE-BLOCK-START" in seed
        assert "EVOLVE-BLOCK-END" in seed
        assert "build_graph" in seed
        assert "Graph()" in seed

    def test_markers_wrap_body_only(self):
        store = ExperienceStore()
        store.add(Experience("gemm", SAMPLE_STEP_CODE, 100.0, 1e-6, {}, []))

        seed = build_seed(store, "gemm", {})
        lines = seed.split("\n")
        def_idx = next(i for i, l in enumerate(lines) if "def build_graph" in l)
        start_idx = next(i for i, l in enumerate(lines) if "EVOLVE-BLOCK-START" in l)
        end_idx = next(i for i, l in enumerate(lines) if "EVOLVE-BLOCK-END" in l)
        assert def_idx < start_idx < end_idx


if __name__ == "__main__":
    unittest.main()
