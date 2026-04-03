import unittest

from src.experience_store import Experience, ExperienceStore
from src.kernel_curator import (
    build_curator_user_prompt,
    parse_curator_response,
)

GEMM_REF = "def forward(A, B): return torch.matmul(A, B)"
RMS_REF = "def forward(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)"
SILU_REF = "def forward(x): return x * torch.sigmoid(x)"


class TestBuildCuratorUserPrompt(unittest.TestCase):
    def test_excludes_target_kernel(self):
        store = ExperienceStore()
        store.add(Experience("gemm", "code", 100.0, 1e-6, {}, ["matmul"]))
        store.add(Experience("rms_norm", "code", 50.0, 1e-6, {}, ["reduction"]))

        prompt = build_curator_user_prompt(
            store, "gemm", GEMM_REF,
            {"rms_norm": RMS_REF},
        )
        assert "rms_norm" in prompt
        # gemm should NOT appear as a candidate heading
        assert "### gemm" not in prompt

    def test_empty_store_returns_empty(self):
        store = ExperienceStore()
        prompt = build_curator_user_prompt(store, "gemm", GEMM_REF, {})
        assert prompt == ""

    def test_includes_tags(self):
        store = ExperienceStore()
        store.add(Experience("gemm", "code", 100.0, 1e-6, {}, ["matmul", "tiling"]))
        store.add(Experience("silu", "code", 30.0, 1e-6, {}, ["activation"]))

        prompt = build_curator_user_prompt(
            store, "rms_norm", RMS_REF,
            {"gemm": GEMM_REF, "silu": SILU_REF},
        )
        assert "matmul" in prompt
        assert "activation" in prompt

    def test_includes_target_reference(self):
        store = ExperienceStore()
        store.add(Experience("gemm", "code", 100.0, 1e-6, {}, ["matmul"]))

        prompt = build_curator_user_prompt(
            store, "rms_norm", RMS_REF,
            {"gemm": GEMM_REF},
        )
        assert "rsqrt" in prompt  # from target reference
        assert "Target Kernel: rms_norm" in prompt

    def test_includes_candidate_references(self):
        store = ExperienceStore()
        store.add(Experience("gemm", "code", 100.0, 1e-6, {}, ["matmul"]))
        store.add(Experience("silu", "code", 30.0, 1e-6, {}, ["activation"]))

        prompt = build_curator_user_prompt(
            store, "rms_norm", RMS_REF,
            {"gemm": GEMM_REF, "silu": SILU_REF},
        )
        assert "torch.matmul" in prompt  # from gemm reference
        assert "torch.sigmoid" in prompt  # from silu reference


class TestParseCuratorResponse(unittest.TestCase):
    def test_parses_comma_separated(self):
        available = ["gemm", "rms_norm", "silu_activation"]
        result = parse_curator_response("gemm, rms_norm", available)
        assert result == ["gemm", "rms_norm"]

    def test_handles_none(self):
        result = parse_curator_response("none", ["gemm"])
        assert result == []

    def test_filters_invalid_names(self):
        available = ["gemm", "rms_norm"]
        result = parse_curator_response("gemm, nonexistent, rms_norm", available)
        assert result == ["gemm", "rms_norm"]

    def test_case_insensitive(self):
        available = ["gemm", "rms_norm"]
        result = parse_curator_response("GEMM, RMS_NORM", available)
        assert result == ["gemm", "rms_norm"]


if __name__ == "__main__":
    unittest.main()
