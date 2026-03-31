#!/usr/bin/env python3
"""StepSandbox: validate and profile STeP IR kernels against reference implementations.

Usage:
    python3 run.py gemm my_attempt.py --preset small
    python3 run.py gemm my_attempt.py --preset small --symbolic
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

# --- Auto-resolve step_artifact sources ---
STEP_SRC = os.environ.get("STEP_ARTIFACT_SRC")
if not STEP_SRC:
    _cached = Path.home() / ".cache" / "acceloptstep" / "step_artifact_src"
    _local = Path(__file__).resolve().parent.parent / "step_artifact" / "src"
    if _cached.is_dir():
        STEP_SRC = str(_cached)
    elif _local.is_dir():
        STEP_SRC = str(_local)
    else:
        sys.exit(
            "Cannot find step_artifact sources. Set STEP_ARTIFACT_SRC or ensure "
            "~/.cache/acceloptstep/step_artifact_src/ exists."
        )
for subdir in ["proto", "sim", "step_py", ""]:
    p = os.path.join(STEP_SRC, subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import importlib.util
import json
import tempfile

import torch
import yaml
import sympy
from networkx import MultiDiGraph

SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
RTOL = 1e-3
ATOL = 1e-3


def load_config():
    with open(os.path.join(SANDBOX_DIR, "sandbox_config.yaml")) as f:
        return yaml.safe_load(f)


def load_machine_config(preset="default"):
    with open(os.path.join(SANDBOX_DIR, "machine_config.yaml")) as f:
        raw = yaml.safe_load(f)
    assert preset in raw, f"Unknown machine config preset: {preset}"
    return raw[preset]


def load_module(path, name="mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def symbolic_profile(graph):
    """Traverse graph nodes, sum off-chip traffic and on-chip requirement."""
    from step_py.ops import StepOps

    total_off_chip = sympy.Integer(0)
    total_on_chip = sympy.Integer(0)

    for node in graph.nodes():
        if isinstance(node, StepOps):
            total_off_chip = sympy.Add(total_off_chip, node.off_chip_traffic())
            total_on_chip = sympy.Add(total_on_chip, node.on_chip_requirement())

    off_chip_val = float(total_off_chip) if total_off_chip.is_number else str(total_off_chip)
    on_chip_val = float(total_on_chip) if total_on_chip.is_number else str(total_on_chip)

    total_compute_bw_used = sum(
        node.compute_bw for node in graph.nodes() if hasattr(node, "compute_bw")
    )

    return {
        "off_chip_bytes": off_chip_val,
        "on_chip_bytes": on_chip_val,
        "total_compute_bw_used": total_compute_bw_used,
    }


def cycle_accurate_profile(graph, output_op, problem_mod, dims):
    """Run cycle-accurate simulator and check correctness against compute_gold()."""
    from sim import simulate, HBMConfig, SimConfig
    from utils.gold_checking import reconstruct_numpy

    mc = load_machine_config()
    hbm = HBMConfig(**mc["hbm"])
    sim = SimConfig(
        channel_depth=mc["sim"]["channel_depth"],
        functional_sim=mc["sim"]["functional_sim"],
        mock_bf16=mc["sim"]["mock_bf16"],
    )

    orig_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="step_sandbox_")
    pb_path = os.path.join(tmpdir, "graph.pb")

    os.chdir(tmpdir)
    cycles, duration_ms, duration_s = simulate(
        graph,
        logging=False,
        hbm_config=hbm,
        sim_config=sim,
        protobuf_file=pb_path,
        db_name=None,
    )

    result = {"cycles": cycles, "duration_ms": duration_ms}

    store_name = output_op.store_file_name
    assert os.path.exists(f"{store_name}.json") and os.path.exists(f"{store_name}.npy"), (
        f"Simulation did not produce output files in {tmpdir}"
    )

    sim_output = reconstruct_numpy(store_name, delete_npy=False)
    os.chdir(orig_dir)

    sim_tensor = torch.from_numpy(sim_output).float()
    gold = problem_mod.compute_gold(dims).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}"
    )

    while sim_tensor.ndim < gold.ndim:
        sim_tensor = sim_tensor.unsqueeze(0)
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

    result["correct"] = passed
    result["max_diff"] = max_diff
    return result


def run_kernel(problem_name, kernel_name, dims, symbolic_only=False):
    """Profile a single kernel against its problem reference."""
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
    assert hasattr(kernel_mod, "build_graph"), f"Kernel must define build_graph(dims)"

    graph, output_op = kernel_mod.build_graph(dims)

    # Symbolic profiling (always runs)
    sym = symbolic_profile(graph)

    mc = load_machine_config()
    bw_ok = sym["total_compute_bw_used"] <= mc["total_compute_bw"]
    on_chip_ok = (
        isinstance(sym["on_chip_bytes"], (int, float))
        and sym["on_chip_bytes"] <= mc["on_chip_memory_bytes"]
    )

    def fmt_bytes(b):
        if isinstance(b, str):
            return b
        if b >= 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MB"
        return f"{b / 1024:.1f} KB"

    print(f"[{label}] Off-chip traffic: {fmt_bytes(sym['off_chip_bytes'])}")
    print(f"[{label}] On-chip memory:   {fmt_bytes(sym['on_chip_bytes'])}")
    print(f"[{label}] Compute BW:       {sym['total_compute_bw_used']}/{mc['total_compute_bw']}"
          f" {'OK' if bw_ok else 'EXCEEDED'}")
    print(f"[{label}] On-chip limit:    {'OK' if on_chip_ok else 'EXCEEDED'}")

    if symbolic_only:
        return sym

    # HW constraint check before running expensive simulation
    assert bw_ok, f"Compute BW exceeded: {sym['total_compute_bw_used']} > {mc['total_compute_bw']}"
    assert on_chip_ok, f"On-chip memory exceeded: {sym['on_chip_bytes']} > {mc['on_chip_memory_bytes']}"

    # Cycle-accurate profiling
    result = cycle_accurate_profile(graph, output_op, problem_mod, dims)

    status = "PASS" if result["correct"] else "FAIL"
    print(f"[{label}] Correctness:      {status} (max_diff={result.get('max_diff', 'N/A'):.2e})")
    print(f"[{label}] Cycles:           {result['cycles']}")
    print(f"[{label}] Duration:         {result['duration_ms']:.3f} ms")

    return {**sym, **result}


def resolve_dims(config, problem_name, preset=None, dims_json=None):
    """Get dims from preset or JSON override."""
    if dims_json:
        return json.loads(dims_json)
    bench = config[problem_name]
    assert preset, f"Must specify --preset or --dims. Available presets: {list(bench['presets'].keys())}"
    assert preset in bench["presets"], (
        f"Unknown preset '{preset}'. Available: {list(bench['presets'].keys())}"
    )
    return bench["presets"][preset]


def list_problems(config):
    """Print available problems and their presets."""
    for name, bench in config.items():
        presets = ", ".join(bench["presets"].keys())
        kernel_dir = os.path.join(SANDBOX_DIR, "kernels", name)
        kernels = []
        if os.path.isdir(kernel_dir):
            kernels = [f for f in os.listdir(kernel_dir) if f.endswith(".py")]
        k_str = ", ".join(kernels) if kernels else "(none)"
        print(f"  {name}: presets=[{presets}] kernels=[{k_str}]")


def main():
    parser = argparse.ArgumentParser(description="StepSandbox: STeP IR kernel profiler")
    parser.add_argument("problem", nargs="?", help="Problem name (e.g. gemm)")
    parser.add_argument("kernel", nargs="?", help="Kernel filename (e.g. my_attempt.py)")
    parser.add_argument("--preset", help="Preset name from sandbox_config.yaml")
    parser.add_argument("--dims", help="JSON dict of dimensions (overrides preset)")
    parser.add_argument("--symbolic", action="store_true", help="Symbolic profiling only (skip simulation)")
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
            run_kernel(args.problem, k, dims, symbolic_only=args.symbolic)
    else:
        assert args.kernel, "Specify a kernel filename or --all"
        dims = resolve_dims(config, args.problem, args.preset, args.dims)
        run_kernel(args.problem, args.kernel, dims, symbolic_only=args.symbolic)


if __name__ == "__main__":
    main()
