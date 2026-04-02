import json
import os
import traceback
from dataclasses import dataclass, asdict

import torch
import numpy as np


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
from math import ceil
import torch
import numpy as np

SEED = 42

from graph.graph import MultiDiGraph as Graph
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
from step_py.kernels.linear import Linear, LinearTileConfig
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
    "functional_sim": False,
    "mock_bf16": False,
}

RTOL = 1e-3
ATOL = 1e-3


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
        from sim import simulate, HBMConfig, SimConfig
        from utils.gold_checking import reconstruct_numpy

        graph, output_op = build_graph(dims)

        hbm = HBMConfig(**HBM_CONFIG)
        sim_cfg = SimConfig(**SIM_CONFIG)

        os.chdir(work_dir)
        pb_path = os.path.join(work_dir, "graph.pb")

        cycles, duration_ms, duration_s = simulate(
            graph,
            logging=False,
            hbm_config=hbm,
            sim_config=sim_cfg,
            protobuf_file=pb_path,
            db_name=None,
        )
        os.chdir(orig_dir)
    except Exception:
        os.chdir(orig_dir)
        result = EvalResult(stage="simulate", success=False, code=code,
                            error_message=f"simulate failed: {traceback.format_exc()}")
        _write_result(result, work_dir)
        return result

    # Stage 3: correctness
    store_name = output_op.store_file_name
    store_path = os.path.join(work_dir, store_name)
    assert os.path.exists(f"{store_path}.json") and os.path.exists(f"{store_path}.npy"), (
        f"Simulation did not produce output files: {store_name}.json / {store_name}.npy"
    )

    sim_output = reconstruct_numpy(store_path, delete_npy=False)
    sim_tensor = torch.from_numpy(sim_output).float()
    gold = problem_module.compute_gold(dims).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}"
    )
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
