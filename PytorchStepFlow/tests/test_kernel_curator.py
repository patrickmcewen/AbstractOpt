from src.agents.kernel_curator import build_curator_user_prompt, parse_curator_response
from src.experience_store import ExperienceStore


def test_build_curator_user_prompt():
    store = ExperienceStore()
    store.add("gemm", "code_a", 500.0)
    store.add("sdpa", "code_b", 2000.0)
    store.add("mlp", "code_c", 1500.0)
    prompt = build_curator_user_prompt(store, target_kernel="sdpa")
    assert "gemm" in prompt
    assert "mlp" in prompt
    assert "sdpa" not in prompt  # exclude target kernel from candidates
    assert "500.0" in prompt


def test_parse_curator_response_valid():
    response = "gemm, mlp"
    selected = parse_curator_response(response, available_kernels=["gemm", "mlp", "layernorm"])
    assert selected == ["gemm", "mlp"]


def test_parse_curator_response_filters_invalid():
    response = "gemm, nonexistent_kernel, mlp"
    selected = parse_curator_response(response, available_kernels=["gemm", "mlp"])
    assert selected == ["gemm", "mlp"]


def test_parse_curator_response_empty():
    response = "none"
    selected = parse_curator_response(response, available_kernels=["gemm", "mlp"])
    assert selected == []
