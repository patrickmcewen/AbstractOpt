import unittest
from pathlib import Path

from src.experience_store import Experience
from src.prompt_builder import build_system_message


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class TestBuildSystemMessage(unittest.TestCase):
    def test_includes_step_rules(self):
        msg = build_system_message(
            kernel_spec="def forward(A, B): return torch.matmul(A, B)",
            curated_experiences=[],
            prompts_dir=str(PROMPTS_DIR),
        )
        assert "STeP IR" in msg
        assert "build_graph" in msg

    def test_includes_kernel_spec(self):
        spec = "class Model(nn.Module):\n    def forward(self, x): return x * 2"
        msg = build_system_message(
            kernel_spec=spec,
            curated_experiences=[],
            prompts_dir=str(PROMPTS_DIR),
        )
        assert spec in msg

    def test_includes_curated_examples(self):
        exps = [
            Experience("gemm", "def build_graph(dims): pass", 100.0, 1e-6, {}, ["matmul"]),
            Experience("rms_norm", "def build_graph(dims): pass", 50.0, 1e-6, {}, ["reduction"]),
        ]
        msg = build_system_message(
            kernel_spec="spec here",
            curated_experiences=exps,
            prompts_dir=str(PROMPTS_DIR),
        )
        assert "gemm" in msg
        assert "rms_norm" in msg
        assert "Reference Implementations" in msg

    def test_no_curated_section_when_empty(self):
        msg = build_system_message(
            kernel_spec="spec",
            curated_experiences=[],
            prompts_dir=str(PROMPTS_DIR),
        )
        assert "Reference Implementations" not in msg


if __name__ == "__main__":
    unittest.main()
