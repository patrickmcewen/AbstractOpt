# Handoff: STeP Analytical Timing Model

## Goal

Build an analytical performance model for STeP dataflow graphs that predicts cycle counts. The model walks the graph topologically computing per-node timing quantities (T_fire, N_fire, NIT, OTI, OCI, ICI, ICD, fto, end), then validates against the cycle-accurate Rust simulator (`step_perf`).

## Current State

The model achieves **4.9% average error across 61 kernel/preset combinations** (with startup=0).

### Recently Fixed: Buffer Absorption Regression on moe_expert_single

Buffer absorption (`pred.end <= st` → zero pred OTI) was firing incorrectly on the `Bufferize_8 → Streamify_9 → Reshape_10` chain in moe_expert_single. The fix: only apply buffer absorption when the predecessor's end time is **symbolic** (has free symbols before substitution). This targets MoE routing patterns where expert paths complete early due to routing decisions, while concrete `pred.end == st` cases (e.g. Bufferize→Streamify) preserve rate info that downstream nodes need.

Additionally, buffer absorption is now **pass2-only** via a `buffer_absorption=False` flag on `_run_timing_pass`, preventing cascading into contention estimation.

Result: moe_expert_single/mixtral_8x went from 21.9% → 2.7%. moe_routed/small preserved at 3.4%.

**Note:** The underlying issue is that Streamify's T_fire is underestimated (model: ~0 cycles, sim: 16,389 cycles for moe_expert_single). The model treats Streamify as a zero-cost passthrough. A future improvement would be to properly model Streamify's PMU read cost.

## What's Implemented

### Recent Changes (this session)

**1. Reshape padding OTI adjustment** (`timing.py:232-240`)

Reshape with `pad_fn` produces tiles in groups of `chunk_size` — some are real (from predecessor) and some are padding (locally generated, zero cost). The model's OTI=OCI assumes all tiles wait for the predecessor, but padding tiles arrive instantly.

Fix: `OTI = OCI / chunk_size` for Reshape with pad_fn. Only applied in pass2 via `adjust_reshape_padding=True` flag (not pass1, where it cascades into contention estimation).

Impact: moe_routed/small 55% → 41% → 3.1% (combined with buffer absorption).

**2. Buffer absorption** (`timing.py:186-228`)

When a predecessor finishes before the consumer starts (`pred.end <= st`), all tiles are buffered. The consumer processes at T_fire rate, not limited by the producer's OTI.

Implementation: `st` is computed before OCI. In the OCI loop, `_is_leq(pred_end, st, sym_subs)` checks if the predecessor finished. If so, `pred_oti = 0`.

Impact: moe_routed/small 41% → 3.1%.

**Fixed:** Buffer absorption is now pass2-only via `buffer_absorption=True` flag, and only fires when pred.end is symbolic (has free symbols). This preserves rate info for concrete Bufferize→Streamify chains while allowing MoE routing absorption.

**3. sym_subs returned from analyze_timing** (`timing.py:408`, `validate_timing.py:131-134`)

`analyze_timing()` now returns the `sym_subs` dict (expected values from FlatPartition: `input_N_fire / num_consumers`) in its result. `validate_timing.py` uses these instead of defaulting to `{s: 1}` for all symbols.

### Pre-existing Implementation

**HBM OTI formula** (per-tile off-chip access time):
```
OTI = dispatch + channel_time + startup + backpressure
    = ceil(R/par_dispatch) + (ceil(concurrent_R/C)-1)*II + latency + startup + 3
```

**Contention**: Two-pass with weighted overlap fractions. Pass 1 estimates intervals without contention. Pass 2 uses overlap-fraction-weighted `concurrent_R`.

**PMU read cost**: `T_fire = max(pmu_read_cycles, comp_cycles, store_cycles)`. Only charged for inputs from PMU-backed sources.

**Accumulating ops**: `OCI = NIT * max(T_fire, per_step_ICI)` — sequential fires, not pipelined.

**FlatPartition fan-out**: `OTI *= num_consumers` (round-robin interleaving).

**FlatReassemble fan-in**: `pred_OTI /= n_data_inputs` (interleaved collection).

## What Worked (Keep These)

1. **Additive HBM model** (not multiplicative) — dispatch + channel_time + overhead
2. **PMU read in roofline** — `max(pmu_read, comp, store)` with `_produces_pmu_tile` tracking
3. **Accumulating OCI** — `NIT * max(T_fire, per_step_ICI)` for sequential fires
4. **Weighted contention** — overlap-fraction weighting prevents short-lived ops inflating long-running ones
5. **FlatPartition/FlatReassemble fan-out/fan-in** — OTI scaling + OTI divisor for MoE patterns
6. **Smart sym_subs** — `input_N_fire / num_consumers` for expected-value substitution
7. **Reshape padding OTI** — `OCI / chunk_size` in pass2 only
8. **Buffer absorption** — zeroing pred OTI when `pred.end <= st` AND pred.end is symbolic. Only targets MoE routing patterns where symbolic N_fire causes early completion.
9. **Pass1/pass2 separation** — adjustments that change intervals (Reshape padding, buffer absorption) should only apply in pass2 to avoid cascading into contention estimation

## What Didn't Work (Don't Repeat)

1. **Multiplicative pipeline_factor = 3** — overestimates large R, underestimates small R
2. **Iterative contention refinement** — converges to same values in 1 iteration
3. **sym_subs = 0** — not the expected value; sym_subs = B/n_experts is principled
4. **FlatPartition OTI scaling alone** — scales OTI by fan_out but NIT multiplies it back up
5. **Binary contention overlap** — treating any overlap as full concurrency massively overestimates
6. **Symbolic OTI from Reshape padding** — using `(real_input_tiles - 1) * oci / (n_fire - 1)` creates complex sympy expressions that cascade into contention intervals and inflate Load_0's OTI from 6 to 28. Use `OCI / chunk_size` instead (concrete, no symbolic cascade).
7. **Applying Reshape padding fix in pass1** — changes contention intervals, causing Load_0 OTI to jump from 6 to 28 due to new overlap between weight loads and initial data loads
8. **Buffer absorption with <= on concrete pred.end** — `pred.end == st` fires for Bufferize→Streamify chain, zeroing OTI that downstream needs. Strict `<` breaks moe_routed (needs `==` for symbolic cases). Fix: only absorb when pred.end is symbolic.
9. **Streamify T_fire = OTPC * ceil(tile_bytes / PMU_BW)** — overestimates for bufferize_roundtrip (110-222% error). The sim's Streamify cost is much higher than PMU read model predicts for some cases and much lower for others.

## Key Files

| File | Purpose |
|---|---|
| `step_tl/src/step_py/timing.py` | **Analytical timing model** (main file) |
| `step_tl/src/step_py/ops.py` | Operator classes with T_fire, N_fire, NIT, OTPC |
| `StepDB/validate_timing.py` | Validation harness (model vs sim) — **currently has startup=0** |
| `StepDB/sim_timing.py` | Detailed per-node comparison with gantt charts |
| `StepDB/bench_config.yaml` | Kernel configs and presets |
| `ANALYTICAL_MODEL.md` | Full model documentation (needs update for recent changes) |

## How to Run

```bash
cd /home/ubuntu/patrick/AbstractOpt/STePTiming/StepDB
source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate

# Small presets only
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python validate_timing.py --all

# One kernel verbose
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python validate_timing.py gemm small -v

# Detailed per-node comparison with gantt chart
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python sim_timing.py moe_routed small

# All presets (61 kernels)
PYTHONPATH="../step_tl/src:../step_tl/src/proto:$PYTHONPATH" python -c "
from validate_timing import validate_kernel, load_config
config = load_config()
results = []
for k in [k for k, v in config.items() if v.get('origin') == 'seed']:
    for p in config[k]['presets']:
        try: results.append(validate_kernel(k, p, config))
        except Exception as e: print(f'  SKIP {k}/{p}: {e}')
for k, p, pred, act, err in results:
    print(f'{k:30s} {p:15s} {pred:>10d} {act:>10d} {err:>7.1f}%')
avg = sum(e for _,_,_,_,e in results) / len(results)
print(f'Average: {avg:.1f}% ({len(results)} kernels)')
"
```

## Next Steps (Priority Order)

### 1. HBM backpressure overestimation for large tiles

With startup=0, copy_2d/large overestimates by 28%, bufferize_roundtrip/large by 29%. The `backpressure_overhead=3` was calibrated on R=16 and doesn't scale to R=64. See original investigation plan in previous HANDOFF version.

### 2. Restore startup=14

After fixing the HBM model, re-enable `per_channel_start_up_time=14` in both model and simulator. Currently zeroed for investigation.

### 3. Update ANALYTICAL_MODEL.md

Document Reshape padding fix, buffer absorption, and sym_subs return.

## Current Results (61 kernels, startup=0)

Average error: **4.9%**.

| Kernel | Avg Error | Notes |
|--------|-----------|-------|
| gemm (4 presets) | 4.0% | |
| gemm_tile_mn (4) | 0.5% | |
| gemm_tile_mnk (4) | 3.9% | |
| matmul_binarymap (4) | 0.4% | |
| element_wise_add (4) | 0.5% | |
| gated_mlp (4) | 1.2% | |
| sdpa_core (5) | 2.3% | |
| qkv_projection (4) | 0.6% | |
| copy_2d (4) | 8.5% | large=28% (backpressure model) |
| silu_activation (4) | 1.2% | |
| rms_norm (4) | 4.5% | |
| bufferize_roundtrip (3) | 13.8% | large=29% |
| moe_expert_single (4) | 7.3% | mixtral_8x=2.7% (fixed) |
| gemm_tile_mk (5) | 15.8% | wide=28% |
| moe_routed (4) | 9.1% | small=3.4% (was 55%!) |

## HBM Simulator Config (current — startup=0 for investigation)

```python
HBMConfig(addr_offset=64, channel_num=32, per_channel_latency=2,
          per_channel_init_interval=2, per_channel_outstanding=1,
          per_channel_start_up_time=0)   # was 14, zeroed for investigation
SimConfig(channel_depth=2, functional_sim=True, mock_bf16=False)
```
