import unittest
import sys
from pathlib import Path

# Ensure openevolve is importable for the test
_OE_PATH = str(Path(__file__).resolve().parent.parent.parent / "openevolve")
if _OE_PATH not in sys.path:
    sys.path.insert(0, _OE_PATH)

from src.config import StepEvolveConfig
from src.orchestrator import build_oe_config


class TestBuildOeConfig(unittest.TestCase):
    def test_returns_openevolve_config(self):
        from openevolve.config import Config

        se_config = StepEvolveConfig(
            kernels=[],
            oe_iterations_per_round=30,
            oe_num_islands=2,
            oe_population_size=100,
            llm_model="test-model",
            llm_api_base="http://test",
        )

        oe_cfg = build_oe_config(se_config, system_message="test message")
        assert isinstance(oe_cfg, Config)
        assert oe_cfg.max_iterations == 30
        assert oe_cfg.database.num_islands == 2
        assert oe_cfg.database.population_size == 100
        assert oe_cfg.evaluator.parallel_evaluations == 1


if __name__ == "__main__":
    unittest.main()
