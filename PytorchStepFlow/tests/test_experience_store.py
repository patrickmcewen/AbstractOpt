from src.experience_store import ExperienceStore
from src.agent_config import Experience


def test_add_and_get_all():
    store = ExperienceStore()
    store.add("gemm", "code_a", 1000.0)
    store.add("gemm", "code_b", 500.0)
    results = store.get_all_for_kernel("gemm")
    assert len(results) == 2
    assert results[0].cycle_time == 500.0  # sorted ascending


def test_get_best_per_kernel():
    store = ExperienceStore()
    store.add("gemm", "code_a", 1000.0)
    store.add("gemm", "code_b", 500.0)
    store.add("sdpa", "code_c", 2000.0)
    best = store.get_best_per_kernel()
    assert best["gemm"].cycle_time == 500.0
    assert best["sdpa"].cycle_time == 2000.0


def test_get_summary():
    store = ExperienceStore()
    store.add("gemm", "code_a", 500.0)
    store.add("sdpa", "code_c", 2000.0)
    summary = store.get_summary()
    assert len(summary) == 2
    assert {"kernel_name": "gemm", "cycle_time": 500.0} in summary
    assert {"kernel_name": "sdpa", "cycle_time": 2000.0} in summary


def test_empty_store():
    store = ExperienceStore()
    assert store.get_all_for_kernel("gemm") == []
    assert store.get_best_per_kernel() == {}
    assert store.get_summary() == []
