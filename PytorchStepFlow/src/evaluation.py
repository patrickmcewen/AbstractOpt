import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict

import torch
import numpy as np


SIM_TIMEOUT_SECONDS = 90
SIM_LOGGING = True


@dataclass
class EvalResult:
    stage: str          # "exec" | "simulate" | "correctness" | "success"
    success: bool
    code: str
    error_message: str | None = None
    cycle_time: float | None = None
    max_diff: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


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
    SubImmediate, Div, Silu, RowWiseSum, Exp, Pow2, Rsqrt, MaskRow,
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
"""


HBM_CONFIG = {
    "addr_offset": 64,
    "channel_num": 32,
    "per_channel_latency": 2,
    "per_channel_init_interval": 2,
    "per_channel_outstanding": 1,
    "per_channel_start_up_time": 14,
}

SIM_CONFIG = {
    "channel_depth": 2,
    "functional_sim": True,
    "mock_bf16": False,
}

RTOL = 1e-3
ATOL = 1e-3


# Subprocess script for running simulation with timeout
_SIM_RUNNER_SCRIPT = '''\
import json, sys, os
os.chdir(sys.argv[1])
from sim import HBMConfig, SimConfig
import step_perf

pb_path = sys.argv[2]
hbm_cfg = json.loads(sys.argv[3])
sim_cfg = json.loads(sys.argv[4])
logging = sys.argv[5] == "True" if len(sys.argv) > 5 else False

hbm = HBMConfig(**hbm_cfg)
sim = SimConfig(**sim_cfg)

ret = step_perf.run_graph(pb_path, logging, hbm, sim, None)
if len(ret) == 4:
    _, cycles, dur_ms, dur_s = ret
elif len(ret) == 2:
    _, cycles = ret
    dur_ms, dur_s = 0.0, 0.0
else:
    raise RuntimeError(f"Unexpected return: {ret}")
print(json.dumps({"cycles": cycles, "dur_ms": dur_ms, "dur_s": dur_s}))
'''


def _strip_imports(code: str) -> str:
    """Remove import lines from LLM-generated code since we prepend our own."""
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


def _write_result(result: EvalResult, work_dir: str):
    with open(os.path.join(work_dir, "result.json"), "w") as f:
        f.write(result.to_json())


def evaluate_kernel(code: str, problem_module, dims: dict, work_dir: str) -> EvalResult:
    """Evaluate an executor-generated kernel against a PyTorch reference.

    Stages: exec -> simulate -> correctness -> success.
    All artifacts are written to work_dir.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Always write the code
    with open(os.path.join(work_dir, "body.py"), "w") as f:
        f.write(code)

    # Stage 1: exec — prepend standard imports so LLM code doesn't need them
    namespace = {}
    full_code = IMPORT_SCAFFOLD + _strip_imports(code)
    with open(os.path.join(work_dir, "full_body.py"), "w") as f:
        f.write(full_code)
    try:
        exec(full_code, namespace)
    except Exception:
        result = EvalResult(stage="exec", success=False, code=code,
                            error_message=f"exec failed: {traceback.format_exc()}")
        _write_result(result, work_dir)
        return result

    build_graph = namespace.get("build_graph")
    if build_graph is None:
        result = EvalResult(stage="exec", success=False, code=code,
                            error_message="Code does not define build_graph")
        _write_result(result, work_dir)
        return result

    # Stage 2: simulate
    orig_dir = os.getcwd()
    try:
        from sim import serialize
        from utils.gold_checking import reconstruct_numpy

        graph, output_op = build_graph(dims)

        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        pb_path = os.path.join(os.getcwd(), "graph.pb")

        # Phase 1: serialize graph to proto (fast, runs in-process)
        serialize(graph, pb_path, SIM_CONFIG["functional_sim"])

        # Phase 2: run simulation in subprocess with timeout
        pythonpath_extra = ":".join([
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # PytorchStepFlow/
            *[p for p in sys.path if "step_tl" in p or "sim" in p or "step_py" in p or "proto" in p],
        ])
        env = os.environ.copy()
        env["PYTHONPATH"] = pythonpath_extra + ":" + os.environ.get("PYTHONPATH", "")
        if SIM_LOGGING:
            env.setdefault("RUST_LOG", "info")

        proc = subprocess.run(
            [sys.executable, "-c", _SIM_RUNNER_SCRIPT,
             os.getcwd(), pb_path,
             json.dumps(HBM_CONFIG), json.dumps(SIM_CONFIG),
             str(SIM_LOGGING)],
            capture_output=True, text=True, timeout=SIM_TIMEOUT_SECONDS,
            env=env,
        )
        sim_stderr = proc.stderr
        assert proc.returncode == 0, f"Simulator failed:\n{sim_stderr[-2000:]}"
        sim_result = json.loads(proc.stdout.strip().split("\n")[-1])
        cycles = sim_result["cycles"]

        os.chdir(orig_dir)
    except subprocess.TimeoutExpired:
        os.chdir(orig_dir)
        result = EvalResult(stage="simulate", success=False, code=code,
                            error_message=f"Simulation timed out after {SIM_TIMEOUT_SECONDS}s")
        _write_result(result, work_dir)
        return result
    except BaseException as exc:
        os.chdir(orig_dir)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        result = EvalResult(stage="simulate", success=False, code=code,
                            error_message=f"simulate failed: {traceback.format_exc()}")
        _write_result(result, work_dir)
        return result

    # Stage 3: correctness
    store_name = output_op.store_file_name
    store_path = os.path.join(work_dir, store_name)
    npy_exists = os.path.exists(f"{store_path}.npy")
    json_exists = os.path.exists(f"{store_path}.json")
    if not npy_exists:
        sim_log = sim_stderr.strip() if sim_stderr else "(no simulator output)"
        result = EvalResult(stage="simulate", success=False, code=code,
                            error_message=(
                                f"Simulation did not produce output file: {store_name}.npy\n"
                                f"Simulator stderr:\n{sim_log[-3000:]}"
                            ))
        _write_result(result, work_dir)
        return result

    if json_exists:
        sim_output = reconstruct_numpy(store_path, delete_npy=False)
    else:
        sim_output = np.load(f"{store_path}.npy")
    sim_tensor = torch.from_numpy(sim_output).float()
    gold = problem_module.compute_gold(dims).float()

    if sim_tensor.numel() != gold.numel():
        result = EvalResult(stage="correctness", success=False, code=code,
                            error_message=f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}",
                            cycle_time=float(cycles))
        _write_result(result, work_dir)
        return result
    while sim_tensor.ndim < gold.ndim:
        sim_tensor = sim_tensor.unsqueeze(0)
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

    if not passed:
        result = EvalResult(stage="correctness", success=False, code=code,
                            error_message=f"Output incorrect: max_diff={max_diff}",
                            cycle_time=float(cycles), max_diff=max_diff)
        _write_result(result, work_dir)
        return result

    # Stage 4: success
    result = EvalResult(stage="success", success=True, code=code,
                        cycle_time=float(cycles), max_diff=max_diff)
    _write_result(result, work_dir)
    return result
