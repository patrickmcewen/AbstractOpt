#!/usr/bin/env python3
"""NKISandbox: validate and profile NKI kernels against reference implementations.

Usage:
    python3 run.py gemm my_attempt.py --preset small
    python3 run.py gemm my_attempt.py --dims '{"M":512, "K":512, "N":512}'
    python3 run.py gemm --all --preset small
    python3 run.py --list
"""

import os
import sys
from pathlib import Path

# --- Auto-bootstrap: re-exec under Neuron venv if needed ---
NEURON_VENV_PYTHON = "/opt/aws_neuronx_venv_pytorch_2_7/bin/python3"
_target_prefix = str(Path(NEURON_VENV_PYTHON).parent.parent)
if sys.prefix != _target_prefix:
    assert Path(NEURON_VENV_PYTHON).exists(), f"Neuron venv not found at {NEURON_VENV_PYTHON}"
    print(f">>> Re-executing under Neuron venv ({NEURON_VENV_PYTHON})...", flush=True)
    os.execv(NEURON_VENV_PYTHON, [NEURON_VENV_PYTHON] + sys.argv)

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
import yaml

SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOL = 2e-5
WARMUP_ITERS = 2
BENCH_ITERS = 10


def load_config():
    with open(os.path.join(SANDBOX_DIR, "sandbox_config.yaml")) as f:
        return yaml.safe_load(f)


def load_module(path, name="mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def l2norm_allclose(actual, expected, rel_tol):
    """Check if L2 norm of difference is within relative tolerance."""
    diff = (actual - expected).astype(np.float64)
    ref_norm = np.linalg.norm(expected.astype(np.float64))
    if ref_norm == 0:
        return np.linalg.norm(diff) < 1e-7, np.linalg.norm(diff)
    rel_diff = np.linalg.norm(diff) / ref_norm
    return rel_diff < rel_tol, rel_diff


def run_kernel(problem_name, kernel_name, dims):
    """Compile, validate, and benchmark an NKI kernel."""
    import neuronxcc.nki as nki

    config = load_config()
    assert problem_name in config, (
        f"Unknown problem: {problem_name}. Available: {list(config.keys())}"
    )

    problem_path = os.path.join(SANDBOX_DIR, config[problem_name]["problem"])
    kernel_path = os.path.join(SANDBOX_DIR, "kernels", problem_name, kernel_name)
    assert os.path.exists(kernel_path), f"Kernel not found: {kernel_path}"

    label = f"{problem_name}/{kernel_name}"
    print(f"[{label}] dims={dims}")

    problem_mod = load_module(problem_path, "problem_mod")
    kernel_mod = load_module(kernel_path, "kernel_mod")

    # Resolve kernel function and inputs
    if hasattr(kernel_mod, "get_nki_kernel"):
        nki_kernel_fn = kernel_mod.get_nki_kernel(dims)
    else:
        assert hasattr(kernel_mod, "nki_kernel"), (
            f"Kernel must define nki_kernel(...) or get_nki_kernel(dims)"
        )
        nki_kernel_fn = kernel_mod.nki_kernel

    assert hasattr(kernel_mod, "get_nki_inputs"), "Kernel must define get_nki_inputs(dims)"
    nki_inputs = kernel_mod.get_nki_inputs(dims)

    # Gold reference
    gold = problem_mod.compute_gold(dims).float().numpy()

    # Compile and run
    baremetal_fn = nki.baremetal(nki_kernel_fn)
    output = baremetal_fn(*nki_inputs)

    # Handle output: may be a single array or tuple
    if isinstance(output, (list, tuple)):
        out_np = output[0]
    else:
        out_np = output
    if isinstance(out_np, torch.Tensor):
        out_np = out_np.numpy()
    out_np = out_np.astype(np.float32)

    # Reshape to match gold if needed
    out_flat = out_np.reshape(-1)
    gold_flat = gold.reshape(-1)
    assert out_flat.shape == gold_flat.shape, (
        f"Shape mismatch: kernel={out_np.shape} gold={gold.shape}"
    )

    passed, rel_diff = l2norm_allclose(out_flat, gold_flat, REL_TOL)
    status = "PASS" if passed else "FAIL"
    max_diff = np.abs(out_flat - gold_flat).max()
    print(f"[{label}] Correctness:  {status} (L2 rel_diff={rel_diff:.2e}, max_abs_diff={max_diff:.2e})")

    # Benchmark latency
    for _ in range(WARMUP_ITERS):
        baremetal_fn(*nki_inputs)

    latencies = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        baremetal_fn(*nki_inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    mean_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    print(f"[{label}] Latency:      {mean_ms:.3f} ms (min={min_ms:.3f}, max={max_ms:.3f})")

    return {
        "correct": passed,
        "rel_diff": rel_diff,
        "max_diff": float(max_diff),
        "latency_ms": float(mean_ms),
        "min_ms": float(min_ms),
        "max_ms": float(max_ms),
    }


def resolve_dims(config, problem_name, preset=None, dims_json=None):
    if dims_json:
        return json.loads(dims_json)
    bench = config[problem_name]
    assert preset, f"Must specify --preset or --dims. Available presets: {list(bench['presets'].keys())}"
    assert preset in bench["presets"], (
        f"Unknown preset '{preset}'. Available: {list(bench['presets'].keys())}"
    )
    return bench["presets"][preset]


def list_problems(config):
    for name, bench in config.items():
        presets = ", ".join(bench["presets"].keys())
        kernel_dir = os.path.join(SANDBOX_DIR, "kernels", name)
        kernels = []
        if os.path.isdir(kernel_dir):
            kernels = [f for f in os.listdir(kernel_dir) if f.endswith(".py")]
        k_str = ", ".join(kernels) if kernels else "(none)"
        print(f"  {name}: presets=[{presets}] kernels=[{k_str}]")


def main():
    parser = argparse.ArgumentParser(description="NKISandbox: NKI kernel profiler")
    parser.add_argument("problem", nargs="?", help="Problem name (e.g. gemm)")
    parser.add_argument("kernel", nargs="?", help="Kernel filename (e.g. my_attempt.py)")
    parser.add_argument("--preset", help="Preset name from sandbox_config.yaml")
    parser.add_argument("--dims", help="JSON dict of dimensions (overrides preset)")
    parser.add_argument("--all", action="store_true", help="Run all kernels for the given problem")
    parser.add_argument("--list", action="store_true", help="List available problems and presets")
    args = parser.parse_args()

    config = load_config()

    if args.list:
        print("Available problems:")
        list_problems(config)
        return

    assert args.problem, "Specify a problem name or --list"

    if args.all:
        dims = resolve_dims(config, args.problem, args.preset, args.dims)
        kernel_dir = os.path.join(SANDBOX_DIR, "kernels", args.problem)
        assert os.path.isdir(kernel_dir), f"No kernels directory: {kernel_dir}"
        kernels = sorted(f for f in os.listdir(kernel_dir) if f.endswith(".py"))
        assert kernels, f"No .py files in {kernel_dir}"
        for k in kernels:
            print(f"\n--- {args.problem}/{k} ---")
            run_kernel(args.problem, k, dims)
    else:
        assert args.kernel, "Specify a kernel filename or --all"
        dims = resolve_dims(config, args.problem, args.preset, args.dims)
        run_kernel(args.problem, args.kernel, dims)


if __name__ == "__main__":
    main()
