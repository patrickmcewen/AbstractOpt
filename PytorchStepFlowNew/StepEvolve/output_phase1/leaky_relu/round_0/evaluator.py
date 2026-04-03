"""Auto-generated evaluator script for leaky_relu / small."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

# --- Embedded configuration ---
STEPDB_PATH = '/home/ubuntu/patrick/AbstractOpt/PytorchStepFlowNew/StepDB'
STEP_TL_SRC = '/home/ubuntu/patrick/AbstractOpt/PytorchStepFlowNew/step_tl/src'
STEP_TL_PROTO = '/home/ubuntu/patrick/AbstractOpt/PytorchStepFlowNew/step_tl/src/proto'
REFERENCE_PATH = '/home/ubuntu/patrick/AbstractOpt/PytorchStepFlowNew/KernelLib/kernels/leaky_relu/reference.py'
DIMS = {'M': 32, 'K': 64, 'negative_slope': 0.01}
KERNEL_NAME = 'leaky_relu'
PRESET = 'small'

IMPORT_SCAFFOLD = 'import math\nfrom math import ceil\nimport torch\nimport numpy as np\n\nSEED = 42\n\nfrom graph.graph import MultiDiGraph as Graph\nfrom rewrite.broadcast import infer_broadcast\nfrom step_py.datatype import (\n    Float32, Float16, Uint32, Uint64, Bool,\n    Tile, DynTile, Buffer, MultiHot, Index, Stream,\n)\nfrom step_py.dyndim import DynDim\nfrom step_py.ops import (\n    LinearOffChipLoad, LinearOffChipLoadRef, DynLinearOffChipLoad,\n    RandomOffChipLoad, RandomOffChipStore,\n    OffChipStore, DynOffChipStore,\n    BinaryMap, UnaryMap, BinaryMapAccum, Accum,\n    Promote, PromoteOuter, ExpandRef, RepeatRef, RepeatStatic,\n    Flatten, Reshape, ReshapePadStream,\n    Bufferize, Streamify, DynStreamify,\n    Broadcast, Parallelize, StaticReassemble,\n    FlatPartition, FlatReassemble, EagerMerge,\n    RetileStreamify, FlatmapFilterRowStreamify, FlatmapCounter,\n    MockStreamOp,\n)\nfrom step_py.utility_ops import (\n    SelectGen, ExpertAddrGen, MetadataGen, CacheReadAddrGen,\n    FilterLastTile, PrinterContext, ConsumerContext,\n)\nfrom step_py.functions import map_fn, map_accum_fn, accum_fn, init_fn\nfrom step_py.functions.map_fn import (\n    Matmul, DynMatmul, Mul, MulImmediate, IsEqual, Add, AddImmediate,\n    SubImmediate, Div, Silu, RowWiseSum, Exp, Pow2, Rsqrt, Square, MaskRow,\n    SetOffset, RowWiseAppend, CacheWriteAddrGen, SelectToScalar, ToConstInt,\n)\nfrom step_py.functions.map_accum_fn import (\n    Matmul as MapAccumMatmul, DynMatmul as MapAccumDynMatmul,\n)\nfrom step_py.functions.accum_fn import (\n    Mul as AccumMul, Add as AccumAdd,\n    RetileRow, RetileCol, SignalReqAllRead,\n)\nfrom step_py.functions.init_fn import Zero, Empty, DynEmpty\nfrom step_py.kernels.linear import Linear, LinearTileConfig\n'

SIM_TIMEOUT_SECONDS = 90
RTOL = 1e-3
ATOL = 1e-3


def _setup_paths():
    """Ensure STeP imports resolve."""
    sys.path = [p for p in sys.path if "PytorchStepFlow/" not in p or "PytorchStepFlowNew" in p]
    if STEP_TL_SRC not in sys.path:
        sys.path.insert(0, STEP_TL_SRC)
    if STEP_TL_PROTO not in sys.path:
        sys.path.append(STEP_TL_PROTO)


def _load_reference():
    """Import the reference module."""
    spec = importlib.util.spec_from_file_location("reference", REFERENCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_imports(code: str) -> str:
    """Remove import lines from step_impl code since we prepend our own."""
    lines = code.split("\n")
    result = []
    in_multiline_import = False
    for line in lines:
        stripped = line.strip()
        if in_multiline_import:
            if ")" in stripped:
                in_multiline_import = False
            continue
        if stripped.startswith(("import ", "from ")) and "(" in stripped and ")" not in stripped:
            in_multiline_import = True
            continue
        if stripped.startswith(("import ", "from ")):
            continue
        result.append(line)
    return "\n".join(result)


def _exec_program(program_path):
    """Load and exec the STeP program, return the namespace."""
    _setup_paths()
    code = Path(program_path).read_text()
    full_code = IMPORT_SCAFFOLD + _strip_imports(code)
    namespace = {}
    exec(full_code, namespace)
    return namespace, full_code


def evaluate_stage1(program_path):
    """Stage 1: exec — verify that the program defines build_graph and it runs."""
    _setup_paths()
    namespace, full_code = _exec_program(program_path)
    build_graph = namespace.get("build_graph")
    assert build_graph is not None, "Program does not define build_graph"

    graph, output_op = build_graph(DIMS)
    assert graph is not None, "build_graph returned None graph"
    assert output_op is not None, "build_graph returned None output_op"

    return {"combined_score": 0.3, "stage": "exec", "success": True}


def evaluate_stage2(program_path):
    """Stage 2: simulate + correctness."""
    _setup_paths()
    namespace, full_code = _exec_program(program_path)
    build_graph = namespace.get("build_graph")
    assert build_graph is not None, "Program does not define build_graph"

    work_dir = str(Path(program_path).parent / f"_work_{PRESET}")
    os.makedirs(work_dir, exist_ok=True)

    (Path(work_dir) / "full_body.py").write_text(full_code)

    graph, output_op = build_graph(DIMS)

    from sim import serialize, SimConfig, HBMConfig
    from utils.gold_checking import reconstruct_numpy

    orig_dir = os.getcwd()
    os.chdir(work_dir)
    pb_path = os.path.join(os.getcwd(), "graph.pb")

    sim_config = SimConfig(channel_depth=2, functional_sim=True, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=14,
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

    if proc.returncode != 0:
        return {
            "combined_score": 0.2,
            "stage": "simulate",
            "success": False,
            "error": f"Simulator failed (rc={proc.returncode}): {proc.stderr[-2000:]}",
        }

    sim_result = json.loads(proc.stdout.strip().split("\n")[-1])
    cycles = sim_result["cycles"]

    # Correctness check
    store_name = output_op.store_file_name
    store_path = os.path.join(work_dir, store_name)

    assert os.path.exists(f"{store_path}.npy"), (
        f"Simulation did not produce {store_name}.npy"
    )

    if os.path.exists(f"{store_path}.json"):
        sim_output = reconstruct_numpy(store_path, delete_npy=False)
    else:
        sim_output = np.load(f"{store_path}.npy")

    sim_tensor = torch.from_numpy(sim_output).float()
    ref_mod = _load_reference()
    gold = ref_mod.compute_gold(DIMS).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}"
    )
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

    if not passed:
        return {
            "combined_score": 0.5,
            "stage": "correctness",
            "success": False,
            "cycles": float(cycles),
            "max_diff": max_diff,
        }

    # Score: higher is better, inversely proportional to cycles
    score = 1.0 / (1.0 + float(cycles) / 1000.0)
    return {
        "combined_score": score,
        "stage": "success",
        "success": True,
        "cycles": float(cycles),
        "max_diff": max_diff,
    }


def evaluate(program_path):
    """Full evaluation: stage1 then stage2."""
    s1 = evaluate_stage1(program_path)
    if not s1.get("success", False):
        return s1
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <program_path>"
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
