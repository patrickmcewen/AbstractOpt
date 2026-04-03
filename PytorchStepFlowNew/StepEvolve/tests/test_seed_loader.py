import unittest
from pathlib import Path

from src.experience_store import ExperienceStore
from src.seed_loader import seed_from_stepdb, extract_op_tags

STEPDB_PATH = Path(__file__).resolve().parent.parent.parent / "StepDB"


class TestExtractOpTags(unittest.TestCase):
    def test_detects_matmul(self):
        code = "matmul = BinaryMapAccum(graph=g, in1=a, in2=b, fn=map_accum_fn.Matmul())"
        tags = extract_op_tags(code)
        assert "matmul" in tags

    def test_detects_reduction(self):
        code = "row_sum = UnaryMap(graph=g, input=x, fn=RowWiseSum())"
        tags = extract_op_tags(code)
        assert "reduction" in tags

    def test_detects_element_wise(self):
        code = "added = BinaryMap(graph=g, in1=a, in2=b, fn=Mul())"
        tags = extract_op_tags(code)
        assert "element_wise" in tags

    def test_detects_unary(self):
        code = "act = UnaryMap(graph=g, input=x, fn=Silu())"
        tags = extract_op_tags(code)
        assert "activation" in tags


class TestSeedFromStepDB(unittest.TestCase):
    def test_loads_kernels(self):
        if not STEPDB_PATH.exists():
            self.skipTest("StepDB not found")

        store = ExperienceStore()
        seed_from_stepdb(store, str(STEPDB_PATH))

        summary = store.get_summary()
        assert len(summary) > 0

        for entry in summary:
            best = store.get_best(entry["kernel_name"])
            assert best is not None
            assert "build_graph" in best.code
            assert isinstance(best.tags, list)

    def test_loads_gemm(self):
        if not STEPDB_PATH.exists():
            self.skipTest("StepDB not found")

        store = ExperienceStore()
        seed_from_stepdb(store, str(STEPDB_PATH))

        gemm = store.get_best("gemm")
        assert gemm is not None
        assert "matmul" in gemm.tags
        assert gemm.cycle_time > 0


if __name__ == "__main__":
    unittest.main()
