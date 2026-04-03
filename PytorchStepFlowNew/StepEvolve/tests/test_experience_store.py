import json
import tempfile
import unittest
from pathlib import Path

from src.experience_store import Experience, ExperienceStore


class TestExperience(unittest.TestCase):
    def test_experience_fields(self):
        exp = Experience(
            kernel_name="gemm",
            code="def build_graph(dims): pass",
            cycle_time=100.0,
            max_diff=1e-6,
            dims={"M": 32, "K": 48, "N": 64},
            tags=["matmul", "tiling"],
        )
        assert exp.kernel_name == "gemm"
        assert exp.cycle_time == 100.0
        assert exp.tags == ["matmul", "tiling"]

    def test_experience_to_dict_roundtrip(self):
        exp = Experience(
            kernel_name="rms_norm",
            code="code here",
            cycle_time=50.0,
            max_diff=0.001,
            dims={"M": 64},
            tags=["reduction"],
        )
        d = exp.to_dict()
        exp2 = Experience.from_dict(d)
        assert exp2.kernel_name == exp.kernel_name
        assert exp2.code == exp.code
        assert exp2.cycle_time == exp.cycle_time


class TestExperienceStore(unittest.TestCase):
    def setUp(self):
        self.store = ExperienceStore()

    def test_add_and_get_best(self):
        self.store.add(Experience("gemm", "code1", 100.0, 1e-6, {}, []))
        self.store.add(Experience("gemm", "code2", 50.0, 1e-6, {}, []))
        best = self.store.get_best("gemm")
        assert best.cycle_time == 50.0
        assert best.code == "code2"

    def test_get_best_returns_none_for_unknown(self):
        assert self.store.get_best("nonexistent") is None

    def test_get_all_for_kernel(self):
        self.store.add(Experience("gemm", "c1", 100.0, 1e-6, {}, []))
        self.store.add(Experience("gemm", "c2", 50.0, 1e-6, {}, []))
        self.store.add(Experience("rms_norm", "c3", 30.0, 1e-6, {}, []))
        all_gemm = self.store.get_all_for_kernel("gemm")
        assert len(all_gemm) == 2
        assert all_gemm[0].cycle_time <= all_gemm[1].cycle_time

    def test_get_best_per_kernel(self):
        self.store.add(Experience("gemm", "c1", 100.0, 1e-6, {}, []))
        self.store.add(Experience("gemm", "c2", 50.0, 1e-6, {}, []))
        self.store.add(Experience("rms_norm", "c3", 30.0, 1e-6, {}, []))
        best_map = self.store.get_best_per_kernel()
        assert best_map["gemm"].cycle_time == 50.0
        assert best_map["rms_norm"].cycle_time == 30.0

    def test_get_summary(self):
        self.store.add(Experience("gemm", "c1", 100.0, 1e-6, {}, ["matmul"]))
        self.store.add(Experience("rms_norm", "c2", 30.0, 1e-6, {}, ["reduction"]))
        summary = self.store.get_summary()
        assert len(summary) == 2
        names = {s["kernel_name"] for s in summary}
        assert names == {"gemm", "rms_norm"}
        for s in summary:
            assert "kernel_name" in s
            assert "cycle_time" in s
            assert "tags" in s

    def test_save_and_load(self):
        self.store.add(Experience("gemm", "c1", 100.0, 1e-6, {"M": 32}, ["matmul"]))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        self.store.save(path)

        store2 = ExperienceStore()
        store2.load(path)
        assert len(store2.get_all_for_kernel("gemm")) == 1
        assert store2.get_best("gemm").code == "c1"
        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
