from src.beam_search import select_top_k
from src.agent_config import Experience


def test_select_top_k_basic():
    exps = [
        Experience("gemm", "c", 3000.0),
        Experience("gemm", "a", 1000.0),
        Experience("gemm", "b", 2000.0),
    ]
    result = select_top_k(exps, k=2)
    assert len(result) == 2
    assert result[0].cycle_time == 1000.0
    assert result[1].cycle_time == 2000.0


def test_select_top_k_fewer_than_k():
    exps = [Experience("gemm", "a", 1000.0)]
    result = select_top_k(exps, k=5)
    assert len(result) == 1


def test_select_top_k_empty():
    result = select_top_k([], k=3)
    assert result == []
