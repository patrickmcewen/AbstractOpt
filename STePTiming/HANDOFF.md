# Handoff: STeP Functional Simulation

## Goal

Add functional (value-level) simulation to the STeP Python frontend so kernel outputs can be verified against PyTorch references without running the Rust cycle-accurate simulator. This completes the emulation stack: analytical timing model (done) + functional model (done).

## Current Progress

### Implementation: COMPLETE

**117/117 kernel+preset validations pass.** All 12 seed kernels across all presets produce correct results vs PyTorch references.

### Key Files

| File | Purpose |
|---|---|
| `step_tl/src/step_py/functional.py` | Functional simulation module — `execute(graph, output_op) -> tensor` |
| `StepDB/validate_functional.py` | Validation harness — compares functional sim vs PyTorch reference |

## Architecture

The simulator walks the graph in topological order, computing a `torch.Tensor` per node:

- **Tensor representation:** `(*stream_shape, tile_row, tile_col)` — stream dims + tile dims
- **Multi-output nodes** (Broadcast, FlatPartition): stored as `list[tensor]`
- **Source nodes:** `LinearOffChipLoad` materializes tiles via stride-to-linear-index mapping
- **Compute ops:** batched PyTorch operations (matmul, element-wise, etc.)
- **BinaryMapAccum:** map (batched matmul) then sum-reduce over accumulation rank dims
- **Accum(RetileRow):** vertically concatenates tiles via reshape
- **FlatPartition/FlatReassemble:** routes tiles based on SelectGen multihot tensors
- **Untiling:** reconstructs 2D output by permuting tile dims into spatial dims

## Design Choices

1. **Tensor-level fidelity** — operates on `torch.Tensor`, respects tiling/stride but doesn't simulate tile-by-tile streaming
2. **PyTorch in/out** — no numpy conversion
3. **Standalone executor** — `step_py/functional.py`, separate from timing model
4. **Stride-aware tile materialization** — `_materialize_tiles()` maps multi-dim indices to linear tile positions
5. **BinaryMapAccum = map then reduce** — per-tile matmul, then sum over accumulation rank
6. **All 12 seed kernels** supported, including MoE routing

## Bug Fixed: rms_norm tile_k < K

The functional sim exposed a graph construction bug in `rms_norm/step_impl.py`: it used `MulImmediate(1/tile_k)` + `RowWiseSum` which only computed correct RMS when `tile_k == K`. Fixed by:
1. Changed scaling to `MulImmediate(1/K)` (full K, not tile_k)
2. Added `Accum(Add, rank=1)` after `RowWiseSum` to sum partial row sums across K//tile_k tiles
3. Added `RepeatStatic(K//tile_k)` before the final multiply to broadcast the rsqrt value

Note: the timing model has higher error (~30%) for the new multi-tile rms_norm presets since the graph now has additional Accum + RepeatStatic nodes. Timing model recalibration is a future task.

## How to Run

```bash
cd /home/ubuntu/patrick/AbstractOpt/STePTiming/StepDB
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python validate_functional.py          # all kernels, first preset
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python validate_functional.py gemm small  # specific
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python validate_functional.py --all       # everything
```

## Supported Op Types

Source: `LinearOffChipLoad`, `LinearOffChipLoadRef`, `SelectGen`, `MetadataGen`, `ExpertAddrGen`
Compute: `UnaryMap`, `BinaryMap`, `BinaryMapAccum`, `Accum`
Structural: `Broadcast`, `RepeatStatic`, `Bufferize`, `Streamify`, `Flatten`, `Reshape`, `RetileStreamify`, `ExpandRef`, `RepeatRef`, `Promote`, `PromoteOuter`
Routing: `FlatPartition`, `FlatReassemble`
Sink: `OffChipStore`, `DynOffChipStore`, `PrinterContext`, `ConsumerContext`

## Previous Work: Analytical Timing Model

The analytical timing model is a separate completed project (4.9% avg error across 61 kernel/preset combinations). See below for reference.

---

<details>
<summary>Previous HANDOFF: Analytical Timing Model (click to expand)</summary>

### Goal
Build an analytical performance model for STeP dataflow graphs that predicts cycle counts.

### Current State
The model achieves **4.9% average error across 61 kernel/preset combinations** (with startup=0).

### Key Files
- `step_tl/src/step_py/timing.py` — Analytical timing model
- `StepDB/validate_timing.py` — Validation harness (model vs sim)
- `StepDB/sim_timing.py` — Detailed per-node comparison with gantt charts

### Remaining Timing Model TODOs
1. HBM backpressure overestimation for large tiles (copy_2d/large 28%, bufferize_roundtrip/large 29%)
2. Restore startup=14 after fixing HBM model
3. Update ANALYTICAL_MODEL.md

</details>
