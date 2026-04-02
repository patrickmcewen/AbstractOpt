import importlib.util
import os
import yaml

from src.loop import load_problem_module, resolve_dims, load_experiment_config


def test_load_experiment_config():
    cfg = load_experiment_config("configs/experiment.yaml")
    assert cfg["iterations"] == 3
    assert len(cfg["kernels"]) == 2
    assert cfg["kernels"][0]["name"] == "gemm"


def test_resolve_dims():
    dims = resolve_dims("gemm", "small")
    assert dims["M"] == 256
    assert dims["K"] == 256
    assert dims["N"] == 256


def test_load_problem_module():
    mod = load_problem_module("StepBench/problems/gemm.py")
    assert hasattr(mod, "compute_gold")
    assert hasattr(mod, "get_inputs")
