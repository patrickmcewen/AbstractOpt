"""Integration test: runs StepEvolve on a single kernel for 1 round, 5 iterations.

Requires:
- STeP environment (step_tl, step_perf)
- ANTHROPIC_API_KEY environment variable
- StepDB directory

Run with: pytest tests/test_integration.py -v --timeout=300
"""

import os
import tempfile
import unittest
import sys
from pathlib import Path

STEPDB_PATH = str(Path(__file__).resolve().parent.parent.parent / "StepDB")
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Ensure openevolve is importable
_OE_PATH = str(Path(__file__).resolve().parent.parent.parent / "openevolve")
if _OE_PATH not in sys.path:
    sys.path.insert(0, _OE_PATH)


@unittest.skipUnless(HAS_API_KEY, "ANTHROPIC_API_KEY not set")
class TestSingleKernelEvolution(unittest.TestCase):
    def test_gemm_small_one_round(self):
        """Run one round of evolution on gemm/small with minimal iterations."""
        from src.config import StepEvolveConfig, KernelTarget
        from src.orchestrator import run
        import asyncio

        with tempfile.TemporaryDirectory() as work_dir:
            config = StepEvolveConfig(
                kernels=[
                    KernelTarget(
                        name="gemm",
                        reference_path=os.path.join(STEPDB_PATH, "kernels/gemm/reference.py"),
                        preset="small",
                        dims={"M": 32, "K": 48, "N": 64, "tile_m": 16, "tile_k": 16, "tile_n": 16},
                    )
                ],
                num_rounds=1,
                oe_iterations_per_round=5,
                stepdb_path=STEPDB_PATH,
                work_dir=work_dir,
                oe_num_islands=1,
                oe_population_size=20,
            )

            store = asyncio.run(run(config))

            # Store should have been seeded from StepDB
            assert len(store.get_summary()) > 0

            # Check that work directory was populated
            gemm_dir = Path(work_dir) / "gemm" / "round_0"
            assert gemm_dir.exists()
            assert (gemm_dir / "seed.py").exists()
            assert (gemm_dir / "evaluator.py").exists()


if __name__ == "__main__":
    unittest.main()
