from src.agents.common import make_client, make_agent, load_prompt, render_prompt
from src.agent_config import ExperimentConfig


def test_make_client():
    cfg = ExperimentConfig(
        url="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        iterations=1, plans_per_kernel=1, implementations_per_plan=1, top_k=1,
    )
    client = make_client(cfg)
    assert client.base_url is not None


def test_make_agent():
    cfg = ExperimentConfig(
        url="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        iterations=1, plans_per_kernel=1, implementations_per_plan=1, top_k=1,
    )
    agent = make_agent(cfg, name="TestAgent", system_prompt="You are a test agent.")
    assert agent.name == "TestAgent"


def test_load_prompt(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Hello {name}, your task is {task}.")
    text = load_prompt(str(p))
    assert "{name}" in text


def test_render_prompt():
    template = "Kernel: {kernel_name}\nCode:\n{code}"
    result = render_prompt(template, kernel_name="gemm", code="def f(): pass")
    assert result == "Kernel: gemm\nCode:\ndef f(): pass"


def test_render_prompt_missing_key():
    template = "Kernel: {kernel_name}\nExtra: {extra}"
    result = render_prompt(template, kernel_name="gemm")
    assert "{extra}" in result  # unmatched placeholders left as-is
