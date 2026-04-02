# tests/test_agent_config.py
from src.agent_config import Experience, BuildError, KernelState, ExperimentConfig


def test_experience_creation():
    exp = Experience(kernel_name="gemm", code="def build_graph(): pass", cycle_time=1000.0)
    assert exp.kernel_name == "gemm"
    assert exp.cycle_time == 1000.0


def test_experience_ordering():
    a = Experience(kernel_name="gemm", code="a", cycle_time=500.0)
    b = Experience(kernel_name="gemm", code="b", cycle_time=1000.0)
    assert sorted([b, a], key=lambda e: e.cycle_time) == [a, b]


def test_build_error_creation():
    err = BuildError(
        kernel_name="sdpa",
        code="def build_graph(): pass",
        error_message="TypeError: bad op",
        diagnosis="Used wrong dtype for accumulator",
    )
    assert err.kernel_name == "sdpa"
    assert "wrong dtype" in err.diagnosis


def test_kernel_state_defaults():
    ks = KernelState(kernel_name="gemm", kernel_spec_path="StepBench/problems/gemm.py")
    assert ks.top_k_candidates == []
    assert ks.curated_experiences == []
    assert ks.diagnosed_errors == []


def test_experiment_config_from_files():
    cfg = ExperimentConfig.from_files(
        llm_config_path="configs/llm.json",
        iterations=5,
        plans_per_kernel=4,
        implementations_per_plan=2,
        top_k=3,
    )
    assert cfg.model == "openai/gpt-oss-120b"
    assert cfg.iterations == 5
    assert cfg.plans_per_kernel == 4
    assert cfg.top_k == 3
