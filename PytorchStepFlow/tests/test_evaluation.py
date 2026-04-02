import json
import os
from src.evaluation import EvalResult, evaluate_kernel


def test_eval_result_success():
    r = EvalResult(stage="success", success=True, code="def f(): pass", cycle_time=1234.0, max_diff=0.001)
    assert r.success is True
    assert r.stage == "success"
    assert r.cycle_time == 1234.0


def test_eval_result_failure():
    r = EvalResult(stage="exec", success=False, code="bad code", error_message="SyntaxError")
    assert r.success is False
    assert r.error_message == "SyntaxError"
    assert r.cycle_time is None


def test_eval_result_to_json():
    r = EvalResult(stage="success", success=True, code="def f(): pass", cycle_time=500.0, max_diff=0.0)
    d = r.to_json()
    parsed = json.loads(d)
    assert parsed["stage"] == "success"
    assert parsed["cycle_time"] == 500.0
    assert parsed["success"] is True


def test_eval_result_to_json_with_none():
    r = EvalResult(stage="exec", success=False, code="x", error_message="err")
    d = r.to_json()
    parsed = json.loads(d)
    assert parsed["cycle_time"] is None
    assert parsed["max_diff"] is None


def test_evaluate_kernel_exec_failure(tmp_path):
    """Code that doesn't define build_graph should fail at exec stage."""
    work_dir = str(tmp_path / "eval_exec_fail")
    os.makedirs(work_dir)
    result = evaluate_kernel(
        code="x = 1 + 1",
        problem_module=None,  # won't reach this
        dims={},
        work_dir=work_dir,
    )
    assert result.stage == "exec"
    assert result.success is False
    assert "build_graph" in result.error_message
    assert os.path.exists(os.path.join(work_dir, "body.py"))


def test_evaluate_kernel_syntax_error(tmp_path):
    """Code with syntax error should fail at exec stage."""
    work_dir = str(tmp_path / "eval_syntax")
    os.makedirs(work_dir)
    result = evaluate_kernel(
        code="def build_graph(dims)\n  return None",  # missing colon
        problem_module=None,
        dims={},
        work_dir=work_dir,
    )
    assert result.stage == "exec"
    assert result.success is False


def test_evaluate_kernel_simulate_failure(tmp_path):
    """build_graph that raises should fail at simulate stage."""
    work_dir = str(tmp_path / "eval_sim_fail")
    os.makedirs(work_dir)
    code = "def build_graph(dims):\n    raise RuntimeError('bad graph')"
    result = evaluate_kernel(
        code=code,
        problem_module=None,
        dims={"M": 64},
        work_dir=work_dir,
    )
    assert result.stage == "simulate"
    assert result.success is False
    assert "bad graph" in result.error_message


def test_evaluate_kernel_writes_result_json(tmp_path):
    """Result should be written to result.json in work_dir."""
    work_dir = str(tmp_path / "eval_result_json")
    os.makedirs(work_dir)
    evaluate_kernel(
        code="x = 1",
        problem_module=None,
        dims={},
        work_dir=work_dir,
    )
    result_path = os.path.join(work_dir, "result.json")
    assert os.path.exists(result_path)
    with open(result_path) as f:
        data = json.load(f)
    assert "stage" in data
    assert "success" in data
