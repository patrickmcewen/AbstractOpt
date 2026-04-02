from src.agents.error_curator import build_error_user_prompt, render_diagnosed_errors
from src.agent_config import BuildError


def test_build_error_user_prompt():
    prompt = build_error_user_prompt(
        kernel_name="gemm",
        code="def build_graph(): return bad_op()",
        error_message="NameError: name 'bad_op' is not defined",
    )
    assert "gemm" in prompt
    assert "bad_op" in prompt
    assert "NameError" in prompt


def test_render_diagnosed_errors_empty():
    result = render_diagnosed_errors([])
    assert result == ""


def test_render_diagnosed_errors():
    errors = [
        BuildError("gemm", "code_a", "TypeError", "Used wrong dtype"),
        BuildError("gemm", "code_b", "ValueError", "Bad tile shape"),
    ]
    result = render_diagnosed_errors(errors)
    assert "wrong dtype" in result
    assert "Bad tile shape" in result
    assert "code_a" not in result  # should not include full code, just diagnosis
