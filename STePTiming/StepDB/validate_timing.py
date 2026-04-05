"""Validate the analytical timing model against the cycle-accurate simulator.

Usage:
    python validate_timing.py                    # all seed kernels, first preset
    python validate_timing.py gemm small         # specific kernel + preset
    python validate_timing.py --all              # all seed kernels, small presets
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import sympy
import yaml

STEPDB_DIR = str(Path(__file__).resolve().parent)
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent.parent / "step_tl" / "src" / "proto")
SIM_TIMEOUT_SECONDS = 90

# Ensure imports work
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

# Standard imports prepended to step_impl code
IMPORT_SCAFFOLD = """\
import math
from math import ceil
import torch
import numpy as np

SEED = 42

from graph.graph import MultiDiGraph as Graph
from rewrite.broadcast import infer_broadcast
from step_py.datatype import (
    Float32, Float16, Uint32, Uint64, Bool,
    Tile, DynTile, Buffer, MultiHot, Index, Stream,
)
from step_py.dyndim import DynDim
from step_py.ops import (
    LinearOffChipLoad, LinearOffChipLoadRef, DynLinearOffChipLoad,
    RandomOffChipLoad, RandomOffChipStore,
    OffChipStore, DynOffChipStore,
    BinaryMap, UnaryMap, BinaryMapAccum, Accum,
    Promote, PromoteOuter, ExpandRef, RepeatRef, RepeatStatic,
    Flatten, Reshape, ReshapePadStream,
    Bufferize, Streamify, DynStreamify,
    Broadcast, Parallelize, StaticReassemble,
    FlatPartition, FlatReassemble, EagerMerge,
    RetileStreamify, FlatmapFilterRowStreamify, FlatmapCounter,
    MockStreamOp,
)
from step_py.utility_ops import (
    SelectGen, ExpertAddrGen, MetadataGen, CacheReadAddrGen,
    FilterLastTile, PrinterContext, ConsumerContext,
)
from step_py.functions import map_fn, map_accum_fn, accum_fn, init_fn
from step_py.functions.map_fn import (
    Matmul, DynMatmul, Mul, MulImmediate, IsEqual, Add, AddImmediate,
    SubImmediate, Div, Silu, RowWiseSum, Exp, Pow2, Rsqrt, Square, MaskRow,
    SetOffset, RowWiseAppend, CacheWriteAddrGen, SelectToScalar, ToConstInt,
)
from step_py.functions.map_accum_fn import (
    Matmul as MapAccumMatmul, DynMatmul as MapAccumDynMatmul,
)
from step_py.functions.accum_fn import (
    Mul as AccumMul, Add as AccumAdd,
    RetileRow, RetileCol, SignalReqAllRead,
)
from step_py.functions.init_fn import Zero, Empty, DynEmpty
from step_py.kernels.linear import Linear, LinearTileConfig
"""


def load_config():
    config_path = os.path.join(STEPDB_DIR, "bench_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _strip_imports(code):
    """Remove import lines from step_impl code since we prepend our own."""
    lines = code.split("\n")
    result = []
    in_multiline = False
    for line in lines:
        s = line.strip()
        if in_multiline:
            if ")" in s:
                in_multiline = False
            continue
        if s.startswith(("import ", "from ")) and s != "":
            if "(" in s and ")" not in s:
                in_multiline = True
            continue
        result.append(line)
    return "\n".join(result)


def build_graph_from_impl(kernel_name, dims, config):
    """Build the STeP graph by exec'ing the step_impl code."""
    impl_path = os.path.join(STEPDB_DIR, config[kernel_name]["step_impl"])
    with open(impl_path) as f:
        impl_code = f.read()

    full_code = IMPORT_SCAFFOLD + "\n" + _strip_imports(impl_code)
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, f"build_graph not found in {impl_path}"
    graph, output_op = namespace["build_graph"](dims)
    return graph, output_op


def run_analytical_model(graph, hw_config=None, sym_subs=None):
    """Run the analytical timing model and return predicted cycles.

    Args:
        sym_subs: dict mapping sympy symbol names to concrete values.
            If None and expression has free symbols, assumes uniform
            distribution (each symbolic dim gets value 1).
    """
    from step_py.timing import analyze_timing
    result = analyze_timing(graph, hw_config)
    total = result["total_cycles"]

    # Substitute symbolic dims if needed
    if total.free_symbols:
        if sym_subs is None:
            # Use expected-value substitutions computed by analyze_timing
            # (derived from FlatPartition input_N_fire / num_consumers)
            sym_subs = result.get("sym_subs") or {s: 1 for s in total.free_symbols}
        total = total.subs(sym_subs)
        # Also substitute in per-node info for consistency
        for nid in result["per_node"]:
            for key in ("end", "fto", "st", "OCI", "OTI", "ICI", "ICD"):
                val = result["per_node"][nid].get(key)
                if val is not None and hasattr(val, 'free_symbols') and val.free_symbols:
                    result["per_node"][nid][key] = val.subs(sym_subs)

    total_val = int(sympy.N(total))
    return total_val, result


def run_simulator(graph, output_op, work_dir):
    """Run the cycle-accurate simulator and return actual cycles."""
    from sim import serialize, SimConfig, HBMConfig

    os.makedirs(work_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(work_dir)
    pb_path = os.path.join(os.getcwd(), "graph.pb")

    sim_config = SimConfig(channel_depth=2, functional_sim=True, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=0,
    )

    serialize(graph, pb_path, sim_config.functional_sim)

    sim_runner_script = (
        "import json, sys, os\n"
        "os.chdir(sys.argv[1])\n"
        "from sim import HBMConfig, SimConfig\n"
        "import step_perf\n"
        "pb_path = sys.argv[2]\n"
        "hbm_cfg = json.loads(sys.argv[3])\n"
        "sim_cfg = json.loads(sys.argv[4])\n"
        "hbm = HBMConfig(**hbm_cfg)\n"
        "sim = SimConfig(**sim_cfg)\n"
        "ret = step_perf.run_graph(pb_path, False, hbm, sim, None)\n"
        "if len(ret) == 4:\n"
        "    _, cycles, dur_ms, dur_s = ret\n"
        "elif len(ret) == 2:\n"
        "    _, cycles = ret\n"
        "    dur_ms, dur_s = 0.0, 0.0\n"
        "else:\n"
        "    raise RuntimeError(f'Unexpected return: {ret}')\n"
        "print(json.dumps({'cycles': cycles, 'dur_ms': dur_ms, 'dur_s': dur_s}))\n"
    )

    pythonpath = STEP_TL_SRC + ":" + STEP_TL_PROTO + ":" + os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    proc = subprocess.run(
        [sys.executable, "-c", sim_runner_script,
         os.getcwd(), pb_path,
         json.dumps(asdict(hbm_config)),
         json.dumps({"channel_depth": sim_config.channel_depth,
                      "functional_sim": sim_config.functional_sim,
                      "mock_bf16": sim_config.mock_bf16})],
        capture_output=True, text=True, timeout=SIM_TIMEOUT_SECONDS,
        env=env,
    )

    os.chdir(orig_dir)

    assert proc.returncode == 0, (
        f"Simulator failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    )

    sim_result = json.loads(proc.stdout.strip().split("\n")[-1])
    cycles = int(sim_result["cycles"])

    # Parse TRACE_EVENT lines from stderr if STEP_TRACE is set
    trace_events = []
    if os.environ.get("STEP_TRACE"):
        for line in proc.stderr.split("\n"):
            if line.startswith("TRACE_EVENT|"):
                parts = line.split("|")
                assert len(parts) == 6, f"Bad trace line: {line}"
                trace_events.append({
                    "name": parts[1],
                    "id": int(parts[2]),
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "is_stop": parts[5] == "true",
                })

    if trace_events:
        return cycles, trace_events
    return cycles


def validate_kernel(kernel_name, preset, config, verbose=False):
    """Validate one kernel+preset. Returns (kernel, preset, predicted, actual, error_pct) or error tuple."""
    dims = dict(config[kernel_name]["presets"][preset])
    print(f"\n{'='*60}")
    print(f"  {kernel_name} / {preset}  dims={dims}")
    print(f"{'='*60}")

    graph, output_op = build_graph_from_impl(kernel_name, dims, config)

    # Analytical model
    predicted, detail = run_analytical_model(graph)
    print(f"  Analytical model: {predicted} cycles")

    if verbose:
        for nid, ninfo in detail["per_node"].items():
            node = ninfo["node"]
            print(f"    {str(node):50s}  st={ninfo['st']}  end={ninfo['end']}  OCI={ninfo['OCI']}  OTI={ninfo['OTI']}")

    # Cycle-accurate simulator
    work_dir = os.path.join(STEPDB_DIR, "seed_kernels", kernel_name, f"_work_timing_{preset}")
    actual = run_simulator(graph, output_op, work_dir)
    print(f"  Cycle-accurate sim: {actual} cycles")

    error_pct = abs(predicted - actual) / max(actual, 1) * 100
    print(f"  Error: {error_pct:.1f}%")

    return kernel_name, preset, predicted, actual, error_pct


def main():
    parser = argparse.ArgumentParser(description="Validate analytical timing model")
    parser.add_argument("kernel", nargs="?", help="Kernel name")
    parser.add_argument("preset", nargs="?", help="Preset name")
    parser.add_argument("--all", action="store_true", help="All seed kernels, small presets")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed_kernels = [k for k, v in config.items() if v.get("origin") == "seed"]
    SMALL_PRESETS = ["small", "tiny", "square"]

    results = []
    skipped = []

    def _try_validate(kernel, preset):
        """Run validation, catching failures from unsupported kernels."""
        try:
            return validate_kernel(kernel, preset, config, args.verbose)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            skipped.append((kernel, preset, str(e)))
            return None

    if args.kernel and args.preset:
        r = _try_validate(args.kernel, args.preset)
        if r:
            results.append(r)
    elif args.kernel:
        for preset in config[args.kernel]["presets"]:
            if preset in SMALL_PRESETS:
                r = _try_validate(args.kernel, preset)
                if r:
                    results.append(r)
    elif args.all:
        for kernel in seed_kernels:
            for preset in config[kernel]["presets"]:
                if preset in SMALL_PRESETS:
                    r = _try_validate(kernel, preset)
                    if r:
                        results.append(r)
    else:
        # Default: all seed kernels, first preset only
        for kernel in seed_kernels:
            presets = list(config[kernel]["presets"].keys())
            r = _try_validate(kernel, presets[0])
            if r:
                results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print(f"  {'Kernel':30s} {'Preset':15s} {'Predicted':>10s} {'Actual':>10s} {'Error%':>8s}")
    print(f"{'='*80}")
    for kernel, preset, predicted, actual, err in results:
        print(f"  {kernel:30s} {preset:15s} {predicted:>10d} {actual:>10d} {err:>7.1f}%")
    if skipped:
        print(f"  --- Skipped {len(skipped)} kernel(s) due to errors ---")
        for kernel, preset, reason in skipped:
            print(f"    {kernel}/{preset}: {reason[:80]}")
    print(f"{'='*80}")

    avg_err = sum(e for _, _, _, _, e in results) / len(results) if results else 0
    print(f"  Average error: {avg_err:.1f}% ({len(results)} kernels, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
