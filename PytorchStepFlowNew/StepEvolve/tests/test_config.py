import tempfile
import unittest
from pathlib import Path

from src.config import StepEvolveConfig, KernelTarget, load_config


class TestKernelTarget(unittest.TestCase):
    def test_fields(self):
        kt = KernelTarget(
            name="gemm",
            reference_path="StepDB/kernels/gemm/reference.py",
            preset="small",
            dims={"M": 32, "K": 48, "N": 64},
        )
        assert kt.name == "gemm"
        assert kt.dims["M"] == 32


class TestStepEvolveConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = StepEvolveConfig(kernels=[])
        assert cfg.num_rounds == 5
        assert cfg.oe_iterations_per_round == 50
        assert cfg.max_curated_examples == 4

    def test_load_from_yaml(self):
        yaml_content = """\
num_rounds: 3
oe_iterations_per_round: 20
llm_model: test-model
kernels:
  - name: gemm
    reference_path: StepDB/kernels/gemm/reference.py
    preset: small
    dims: {M: 32, K: 48, N: 64}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        cfg = load_config(path)
        assert cfg.num_rounds == 3
        assert cfg.oe_iterations_per_round == 20
        assert len(cfg.kernels) == 1
        assert cfg.kernels[0].name == "gemm"
        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
