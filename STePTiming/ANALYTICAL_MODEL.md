# STeP Analytical Timing Model — Current Implementation

This document describes the analytical performance model as currently implemented in
`step_tl/src/step_py/timing.py`, including all modifications from the original specification
in `Step_timing.pdf`.

## 1. Overview

The model predicts the total execution time (in cycles) of a STeP dataflow graph by walking
nodes in topological order, computing per-node timing quantities, and propagating rates
through the graph. The total time is:

```
T_graph = max over leaf nodes of end(n)
```

## 2. Per-Node Quantities (from operator definitions)

Each operator class defines four quantities:

| Symbol | Name | Description |
|--------|------|-------------|
| `T_fire(n)` | Firing time | Processing time per firing (roofline) |
| `N_fire(n)` | Firing count | Total number of firings (may be symbolic) |
| `NIT(n, n')` | Input tile count | Tiles consumed from predecessor n' per firing |
| `OTPC(n)` | Output tiles per chunk | Output tiles produced per firing |

These are defined in `ops.py` and `utility_ops.py` per operator class.

## 3. Firing Time (T_fire) — Modified from Original

### Original (Step_timing.pdf)

```
T_fire(n) = max(Size_in / BW_m, FLOPs / BW_c, Size_out / BW_m)
```

### Current Implementation

The implementation matches the cycle-accurate simulator's per-fire roofline, which differs
from the original in two ways:

**3a. PMU read cost is included in the roofline.**

The simulator computes per-fire time as:
```
roofline_cycles = max(load_cycles, comp_cycles, store_cycles)
```
where `load_cycles = sum(tile_bytes / PMU_BW)` for each input tile read from PMU.

The analytical model mirrors this:
```
T_fire(n) = max(T_fire_base(n), pmu_read_cycles)
```
where `T_fire_base` is the op's own roofline (`max(comp, store)`) and `pmu_read_cycles`
comes from `_get_pmu_read_cycles()`.

**3b. PMU reads only apply to inputs from PMU-backed sources.**

Not all inputs are read from PMU. The simulator sets `read_from_mu = true` only on tiles
stored in PMU. The function `_produces_pmu_tile(node)` determines this:

- **Off-chip loads**: always produce PMU tiles (data arrives from HBM into PMU)
- **Bufferize**: always produces PMU tiles (explicitly writes to PMU)
- **Compute ops with `write_back_mu = True`**: output written to PMU
- **Compute ops with `write_back_mu = False`**: output stays in streaming FIFOs (no PMU cost for consumers)
- **Pass-through ops** (Broadcast, RepeatStatic, Reshape, Flatten, Streamify, etc.): inherit PMU status from their predecessor

**3c. Pass-through ops have T_fire = 0.**

Broadcast, RepeatStatic, Reshape, Flatten, Streamify, and similar structural ops have no
compute and no PMU reads in the simulator. They are identified by having neither
`write_back_mu` nor `compute_bw` attributes. The simulator emits no trace events for these ops.

## 4. Core Timing Equations

Computed per node in topological order.

### 4.1 Input Chunk Interval (ICI)

**Original:**
```
ICI(n) = max over predecessors n' of (NIT(n,n') * OTI(n'))
```

**Current — same for off-chip and accumulating ops. Modified for FlatReassemble (see Section 7).**

### 4.2 Output Chunk Interval (OCI) — Modified for Accumulating Ops

**Original:**
```
OCI(n) = max(T_fire(n), ICI(n))
```

**Current — three cases:**

**Case 1: Off-chip memory ops.**
```
OCI(n) = max(HBM_OTI(n), ICI(n))
```
where `HBM_OTI` is computed by the memory contention model (Section 5). This replaces
T_fire (which is 0 for off-chip ops) with the HBM access time.

**Cases 2 & 3: All compute ops (unified formula).**
```
OCI(n) = max over predecessors n' of (NIT(n,n') * max(T_fire(n), OTI(n')))
```

Each predecessor n' contributes `NIT(n,n')` sequential fires to produce one output chunk.
Each fire is gated by `max(T_fire, OTI(n'))` — either compute or input delivery, whichever
is slower. The output rate is limited by the slowest predecessor.

For non-accumulating ops (NIT=1 for all predecessors), this reduces to
`max(T_fire, max over preds of OTI)` — same as the original.

For accumulating ops (NIT=K), this gives `K * max(T_fire, OTI)` per predecessor. This
differs from the original `max(T_fire, K * OTI)` which treats all K input deliveries as
pipelined with a single T_fire. In reality, accumulating ops fire K times sequentially —
each fire needs one input tile and takes T_fire cycles.

**Calibration evidence:** For a 16x16x16 matmul (BinaryMapAccum, NIT=3), the simulator
fires 3 times at 32 cycles each per output tile. The original formula gave
`max(32, 3*11) = 33`, but the actual output tile cadence is `3 * max(32, 11) = 96`.

### 4.3 Output Tile Interval (OTI)

**Same as original:**
```
OTI(n) = OCI(n) / OTPC(n)
```

### 4.4 Input Chunk Delay (ICD) — Modified for FlatPartition Fan-out

**Original:**
```
ICD(n) = max over predecessors n' of ((NIT(n,n') - 1) * OTI(n'))
```

**Current — same formula, plus a FlatPartition override (see Section 7).**

### 4.5 Start Time (st)

**Original:**
```
st(n) = max over predecessors n' of fto(n')
```

**Current — same.** (The original also has an `L_HBM` term for off-chip edges, which is
not currently implemented — off-chip loads are treated as source nodes with no predecessors.)

### 4.6 First Tile Out (fto)

**Same as original:**
```
fto(n) = st(n) + ICD(n) + T_fire(n)
```

For off-chip memory ops, T_fire is replaced by HBM_OTI:
```
fto(n) = st(n) + ICD(n) + HBM_OTI(n)
```

### 4.7 End Time

**Original:**
```
end(n) = st(n) + N_fire(n) * OCI(n)
```

**Current:**
```
end(n) = fto(n) + (N_fire(n) - 1) * OCI(n)
```

This is equivalent when `fto = st + OCI` (i.e., ICD = 0 and T_fire = OCI), but more
accurate when fto includes a startup transient (ICD > 0).

## 5. HBM Memory Contention Model — Significantly Modified

### Original (Step_timing.pdf)

The original uses a simple shared-bandwidth model:
```
OTI(n) = max(1, R_total / C)
```
where R_total is the aggregate requests from all concurrent off-chip ops and C is the
channel count. All concurrent ops see the same OTI.

### Current Implementation

The current model replaces this with a physics-based per-tile HBM access model calibrated
against per-tile simulator traces.

**5.1 Per-tile HBM OTI formula:**
```
HBM_OTI = dispatch + channel_time + startup + backpressure
```

where:
- `dispatch = ceil(R / par_dispatch)` — address dispatch cycles. The load sends R addresses
  in chunks of `par_dispatch` (typically 4), one chunk per cycle.
- `channel_time = (ceil(concurrent_R / C) - 1) * II + latency` — channel bottleneck.
  All concurrent requests distributed across C=32 channels. The busiest channel gets
  `ceil(concurrent_R / C)` requests, spaced by `II=2` (init interval), plus `latency=2`
  (per-channel response latency).
- `startup` — per-channel startup time. Currently set to 0 for calibration investigation.
  (Was 14, matching the simulator's `per_channel_start_up_time`.)
- `backpressure = 3` — additive overhead from bounded DAM channels (depth=2) between
  operators. Calibrated from per-tile traces: copy_2d R=16 startup=0 shows physics=6, sim=9.
  **Known issue: this constant overestimates for large tiles (R=64). Under investigation.**

**5.2 Two-pass contention with weighted overlap:**

Pass 1 computes OTI without contention (each op alone, `concurrent_R = R`). This produces
estimated `[st, end]` intervals for each off-chip op.

Pass 2 computes contention from interval overlaps using **weighted overlap fractions**:

```
concurrent_R(n) = sum over other off-chip ops o of:
    R(o) * overlap_fraction(n, o)
```

where `overlap_fraction = overlap_duration / my_duration`. This prevents short-lived ops
(e.g., 1-tile expert weight loads) from inflating contention for long-running ops (e.g.,
8-tile initial data loads). Binary overlap would count the full R of a 1-cycle weight load
that overlaps with a 160-cycle initial load, which is physically incorrect.

**5.3 Symbolic dimension handling for contention:**

When off-chip op intervals contain symbolic dimensions (e.g., moe_routed), the model
substitutes expected values to get concrete intervals. For symbols created by FlatPartition,
the expected value is `input_N_fire / num_consumers` (uniform routing assumption). Other
symbols default to 1.

## 6. Hardware Parameters

```python
DEFAULT_HW_CONFIG = {
    "hbm_channels": 32,           # C: number of HBM channels
    "hbm_channel_latency": 2,     # per-channel response latency
    "hbm_init_interval": 2,       # minimum cycles between requests on same channel
    "hbm_addr_offset": 64,        # bytes per HBM burst (R = tile_bytes / 64)
    "hbm_startup": 0,             # per-channel startup (under investigation; was 14)
    "compute_bw": 64,             # FLOPs/cycle (default; operators override via compute_bw attr)
}
PMU_BW = 64  # bytes/cycle for on-chip PMU reads and writes
```

Key hardware facts:
- BinaryMapAccum has `compute_bw = 4096` FLOPs/cycle (systolic array), not the default 64.
- PMU bandwidth is 64 bytes/cycle for both reads and writes.
- The simulator uses `SimConfig(channel_depth=2)` — bounded DAM channels with depth 2.

## 7. MoE-Specific Extensions — New

Two extensions handle the fan-out/fan-in pattern of Mixture-of-Experts routing through
FlatPartition and FlatReassemble.

### 7.1 FlatPartition Fan-out (ICD Override)

FlatPartition dispatches tiles round-robin to N expert consumers. Each consumer's tiles are
interleaved with other consumers' tiles. An accumulating op (NIT > 1) downstream of a
FlatPartition must wait for all its input tiles, but its last tile may not arrive until the
FlatPartition finishes dispatching all tiles to all consumers.

**Implementation:** For accumulating ops (NIT > 1), the model walks the predecessor chain
through pass-through ops to find a FlatPartition ancestor. If found:
```
ICD(n) = max(ICD(n), FlatPartition.end - FlatPartition.fto)
```

This replaces the standard `(NIT-1) * OTI_pred` when the FlatPartition's total dispatch
span is larger, modeling the worst case where this consumer's last tile is the last one
dispatched.

### 7.2 FlatReassemble Fan-in (OTI Divisor)

FlatReassemble collects tiles from N parallel expert paths. The expert outputs arrive
interleaved (each expert contributes tiles independently), so the effective tile arrival
rate at FlatReassemble is N times faster than any single expert's output rate.

**Implementation:** For FlatReassemble nodes with N data inputs:
```
effective_OTI(pred) = OTI(pred) / N
```

This divisor is applied when computing ICI and ICD for FlatReassemble.

## 8. Summary of Changes from Original

| Aspect | Original (Step_timing.pdf) | Current Implementation |
|--------|---------------------------|----------------------|
| T_fire | max(in_BW, comp, out_BW) | max(pmu_read, comp, store) — PMU reads only for PMU-backed inputs |
| Pass-through ops | T_fire = 0 | Same, but explicitly identified by absence of `write_back_mu`/`compute_bw` |
| Compute OCI | max(T_fire, NIT * OTI) | max over preds of NIT * max(T_fire, OTI) — sequential fires, per-pred |
| End time | st + N_fire * OCI | fto + (N_fire - 1) * OCI |
| HBM OTI | max(1, R_total / C) | dispatch + channel_time + startup + backpressure (physics-based) |
| Contention | All-concurrent or overlap | Weighted overlap fractions (time-averaged bandwidth sharing) |
| FlatPartition | Not modeled | ICD override for downstream accumulators |
| FlatReassemble | Not modeled | OTI / N_inputs for interleaved collection |
| Symbolic dims | Left symbolic | Expected-value substitution for contention (FlatPartition-aware) |

## 9. Known Limitations

1. **HBM backpressure constant (3 cycles)** overestimates for large tiles (R=64). Under
   investigation — the overhead may decrease as dispatch time dominates and dispatch/response
   pipeline better.

2. **HBM startup** is currently zeroed. The simulator's `per_channel_start_up_time=14`
   was observed to apply per-tile in traces (not per-channel-once as the name suggests).
   This needs further investigation to model correctly.

3. **moe_routed accuracy (~20-30% error)** limited by data-dependent routing. The model
   uses expected uniform routing for contention intervals but can't predict actual routing
   patterns. The FlatPartition ICD override helps but the Accum's compile-time NIT (max
   tiles per output) exceeds the runtime actual.

4. **gemm_tile_mk underestimates** (~14-25%) for some presets. Root cause not yet investigated.

5. **No downstream backpressure modeling.** The rate-based model propagates rates forward
   (producer → consumer) but doesn't capture bounded-channel backpressure where a slow
   consumer throttles a fast producer. This is partially captured by the `backpressure = 3`
   constant but not modeled per-edge.

## 10. Accuracy (61 kernels, all presets, startup=14)

Average error: **4.4%**

| Kernel | Avg Error | Notes |
|--------|-----------|-------|
| gemm (4 presets) | 2.0% | |
| gemm_tile_mn (4) | 0.6% | |
| gemm_tile_mnk (4) | 1.7% | |
| matmul_binarymap (4) | 0.4% | Including 520K-cycle mixtral at 0.4% |
| element_wise_add (4) | 0.4% | |
| gated_mlp (4) | 1.1% | |
| sdpa_core (5) | 1.4% | |
| qkv_projection (4) | 0.6% | |
| copy_2d (4) | 4.8% | large preset 17% (startup masking) |
| silu_activation (4) | 1.3% | |
| rms_norm (4) | 4.5% | |
| bufferize_roundtrip (3) | 7.5% | large preset 18% |
| moe_expert_single (4) | 7.1% | |
| gemm_tile_mk (5) | 12.9% | wide preset 25% |
| moe_routed (4) | 19.7% | Data-dependent routing |
