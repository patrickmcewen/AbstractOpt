import unittest
from pathlib import Path

from src.evaluator_factory import make_evaluator_script

STEPDB_PATH = Path(__file__).resolve().parent.parent.parent / "StepDB"

# Resolve dims and reference_path the way the orchestrator does
_GEMM_DIMS = {"M": 32, "K": 48, "N": 64, "tile_m": 16, "tile_k": 16, "tile_n": 16}
_GEMM_REF = str(STEPDB_PATH / "kernels" / "gemm" / "reference.py")


def _make_gemm_script():
    return make_evaluator_script(
        kernel_name="gemm",
        preset="small",
        dims=_GEMM_DIMS,
        reference_path=_GEMM_REF,
        stepdb_path=str(STEPDB_PATH),
    )


class TestMakeEvaluatorScript(unittest.TestCase):
    def test_generates_valid_python(self):
        compile(_make_gemm_script(), "<evaluator>", "exec")

    def test_has_evaluate_function(self):
        assert "def evaluate(program_path):" in _make_gemm_script()

    def test_has_cascade_stages(self):
        script = _make_gemm_script()
        assert "def evaluate_stage1(program_path):" in script
        assert "def evaluate_stage2(program_path):" in script

    def test_returns_combined_score(self):
        assert "combined_score" in _make_gemm_script()

    def test_includes_dims(self):
        script = _make_gemm_script()
        assert '"M"' in script or "'M'" in script


if __name__ == "__main__":
    unittest.main()
