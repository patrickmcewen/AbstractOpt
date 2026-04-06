"""Analytical timing model for STeP dataflow graphs.

Implements the rate-based performance model from Step_timing.pdf.
Walks the graph in topological order computing per-node timing quantities,
then returns the total graph execution time in cycles.
"""

import math
import sympy
from networkx import MultiDiGraph, topological_sort
from step_py.ops import (
    StepOps, get_stream,
    Bufferize, Streamify,
    FlatPartition, FlatReassemble,
    Reshape,
)
from step_py.datatype import Tile, DynTile, Buffer

DEFAULT_HW_CONFIG = {
    "hbm_channels": 32,
    "hbm_channel_latency": 2,    # per_channel_latency (per request response)
    "hbm_init_interval": 2,      # per_channel_init_interval
    "hbm_addr_offset": 64,       # bytes per HBM burst
    "hbm_startup": 0,            # per_channel_start_up_time (zeroed for calibration)
    "compute_bw": 64,            # FLOPs/cycle
}

PMU_BW = 64  # bytes/cycle


def _get_tile_bytes(node):
    """Get tile size in bytes for an off-chip memory operator."""
    if hasattr(node, 'tile_row') and hasattr(node, 'tile_col') and hasattr(node, 'n_byte'):
        return node.tile_row * node.tile_col * node.n_byte
    if hasattr(node, 'tile_row') and hasattr(node, 'tile_col'):
        in_stream = get_stream(node._input)
        dtype = in_stream.stream_dtype
        assert isinstance(dtype, (Tile, DynTile)), f"Expected Tile, got {type(dtype)}"
        elem_bytes = dtype.tile_dtype.size_in_bytes()
        return node.tile_row * node.tile_col * int(elem_bytes)
    return 1024


def _produces_pmu_tile(node):
    """Check if a node produces tiles stored in PMU (vs streaming FIFOs).

    The simulator sets read_from_mu=true on tiles from:
    - Off-chip loads (data arrives in PMU from HBM)
    - Bufferize (explicitly stores to PMU)
    - Compute ops with write_back_mu=True (output written to PMU)
    - Pass-through ops (Broadcast, RepeatStatic, etc.) inherit from predecessor

    Compute ops with write_back_mu=False produce tiles in streaming FIFOs.
    """
    if node.is_offchip_memory_op():
        return True
    # Compute ops have write_back_mu — check it directly
    wb = getattr(node, 'write_back_mu', None)
    if wb is True:
        return True
    if wb is False:
        return False
    # No write_back_mu attr → check specific op types
    if isinstance(node, Bufferize):
        return True
    if isinstance(node, Streamify):
        # Streamify reads from PMU buffer but outputs to streaming FIFO.
        return False
    # Other pass-through ops (Broadcast, RepeatStatic, etc.) inherit from predecessor
    try:
        inp = node.input
        pred = inp if isinstance(inp, StepOps) else inp[0]
        return _produces_pmu_tile(pred)
    except (AttributeError, NotImplementedError):
        return False


def _get_pmu_read_cycles(node, graph):
    """Compute PMU read cycles for a node based on which inputs come from PMU.

    Only charges PMU read cost for inputs whose producer wrote to PMU.
    The sim does: load_cycle += div_ceil(tile_bytes, PMU_BW) per input with read_from_mu=true.

    Pass-through ops (no write_back_mu, no compute_bw) forward tile references
    without performing PMU reads — return 0 for them.
    """
    if not hasattr(node, 'write_back_mu') and not hasattr(node, 'compute_bw'):
        return 0
    pmu_read = 0

    for inp in node.input_list:
        inp_node = inp if isinstance(inp, StepOps) else inp[0]
        if _produces_pmu_tile(inp_node):
            stream = get_stream(inp)
            dtype = stream.stream_dtype
            if isinstance(dtype, (Tile, DynTile)):
                pmu_read += math.ceil(int(dtype.size_in_bytes()) / PMU_BW)
    return pmu_read


def _get_par_dispatch(node):
    """Get par_dispatch from an off-chip op, default 1."""
    return getattr(node, 'par_dispatch', 1)



def _compute_hbm_oti(requests, par_dispatch, hw_config, concurrent_requests):
    """Compute steady-state OTI for an off-chip op.

    Models the physical per-tile HBM access time. Dispatch and channel
    processing are pipelined (overlap), so the bottleneck is whichever
    takes longer, plus latency/startup/backpressure overheads.

    Calibrated against per-tile simulator traces (STEP_TRACE=1):
    - copy_2d R=16 startup=0: physics=6, sim=9 → backpressure ~3 cycles
    - copy_2d R=16 startup=14: sim=23 = 9+14 (startup paid per tile)
    """
    channels = hw_config.get("hbm_channels", 32)
    init_interval = hw_config.get("hbm_init_interval", 2)
    channel_latency = hw_config.get("hbm_channel_latency", 2)
    startup = hw_config.get("hbm_startup", 14)

    # Dispatch: load sends R addresses in chunks of par_dispatch, one chunk/cycle
    dispatch = math.ceil(requests / par_dispatch)

    # Channel pipeline: all concurrent requests distributed across C channels.
    # Busiest channel gets ceil(total/C) requests, spaced by init_interval.
    reqs_per_channel = math.ceil(concurrent_requests / channels)
    channel_pipe = (reqs_per_channel - 1) * init_interval

    # Dispatch and channel processing overlap: addresses are fed to channels
    # as they are dispatched, so the bottleneck is whichever takes longer.
    # The final response arrives channel_latency after the last request.
    bottleneck = max(dispatch, channel_pipe) + channel_latency

    # Startup: HBM channels need startup_time before first response.
    # Per-tile traces show this is paid on every tile (blocking per-tile model
    # means channels cool down between tile batches).
    startup_overhead = startup

    # Per-tile sim overhead: the Rust sim's blocking per-tile loop executes
    # three time.tick() calls per tile (send_request_time, read_finish_time,
    # on_chip_snd enqueue time) in linear_offchip_load.rs, adding 3 cycles
    # of bookkeeping overhead to every tile in steady state.
    sim_overhead = 3

    return max(1, bottleneck + startup_overhead + sim_overhead)




def _run_timing_pass(nodes, graph, hw_config, offchip_oti,
                     adjust_reshape_padding=False,
                     sym_subs={}):
    """Run one pass of the timing model with given off-chip OTI values."""
    info = {}

    for n in nodes:
        nid = n.instance_id
        preds = list(graph.predecessors(n))

        n_fire = n.N_fire()
        otpc = n.OTPC()
        nit_map = n.NIT()

        # T_fire from the op = max(comp_cycles, store_cycles)
        # PMU read cost added: sim does max(load_cycles, comp_cycles, store_cycles).
        # Pass-through ops (Broadcast, etc.) return T_fire=0 and pmu_read=0.
        t_fire_base = n.T_fire(hw_config)
        pmu_read_cycles = _get_pmu_read_cycles(n, graph)
        t_fire = sympy.Max(t_fire_base, sympy.Integer(pmu_read_cycles))

        # FlatReassemble fan-in: collects tiles from N parallel expert paths.
        # Expert outputs arrive interleaved (round-robin), so the effective
        # per-predecessor OTI is divided by the number of data inputs.
        fan_in_divisor = sympy.Integer(1)
        if isinstance(n, FlatReassemble) and hasattr(n, '_inputs') and len(n._inputs) > 1:
            fan_in_divisor = sympy.Integer(len(n._inputs))

        # Start time — computed before OCI so we can check buffer absorption.
        if not preds:
            st = sympy.Integer(0)
        else:
            st = sympy.Integer(0)
            for pred in preds:
                pid = pred.instance_id
                assert pid in info, f"Predecessor {pred} not yet processed"
                st = sympy.Max(st, info[pid]["fto"])

        # OCI computation
        # Unified formula for compute ops (both accumulating and non-accumulating):
        #   OCI(n) = max over preds n' of (NIT(n,n') * max(T_fire, OTI(n')))
        # Each predecessor contributes NIT sequential fires, each gated by
        # max(T_fire, input_tile_interval). The output rate is limited by the
        # slowest predecessor. For non-accumulating ops (NIT=1 for all preds),
        # this reduces to max(T_fire, max(OTI(preds))).
        if n.is_offchip_memory_op():
            # Off-chip: no compute, rate limited by HBM and input availability
            ici = sympy.Integer(0)
            for pred in preds:
                pid = pred.instance_id
                pred_oti = info[pid]["OTI"]
                nit_count = nit_map.get(pid, sympy.Integer(1))
                ici = sympy.Max(ici, nit_count * pred_oti)
            hbm_oti = sympy.Integer(offchip_oti[nid])
            oci = sympy.Max(hbm_oti, ici)
        else:
            oci = sympy.Integer(0)
            for pred in preds:
                pid = pred.instance_id
                pred_oti = info[pid]["OTI"] / fan_in_divisor
                nit_count = nit_map.get(pid, sympy.Integer(1))
                oci = sympy.Max(oci, nit_count * sympy.Max(t_fire, pred_oti))
            oci = sympy.Max(t_fire, oci)

        oti = oci / otpc

        # Reshape with padding: padding tiles are generated locally at zero
        # cost (no waiting for predecessor). Within each chunk of chunk_size
        # tiles, some may be padding, so the effective per-tile OTI is lower.
        # Approximation: OTI = OCI / chunk_size (uniform padding distribution).
        if adjust_reshape_padding and isinstance(n, Reshape) and n.pad_fn is not None:
            oti = oci / sympy.Integer(n.chunk_size)

        # FlatPartition fan-out: tiles are distributed round-robin to N consumers.
        # Each consumer sees tiles at N times the interval.
        if isinstance(n, FlatPartition):
            oti = oti * sympy.Integer(n.num_consumers)

        # ICD = max over preds of ((NIT - 1) * OTI(pred))
        # Absorbed PMU preds are skipped — no initial collection delay from PMU.
        icd = sympy.Integer(0)
        for pred in preds:
            pid = pred.instance_id
            pred_oti = info[pid]["OTI"] / fan_in_divisor
            nit_count = nit_map.get(pid, sympy.Integer(1))
            icd = sympy.Max(icd, (nit_count - 1) * sympy.Max(t_fire, pred_oti))

        # for off chip ops, t_fire=0. data rate decoupled from latency of sending data to/from HBM,
        # unlike compute ops where tile processing must complete before output tile is produced and next inputs can be consumed.
        if n.is_offchip_memory_op():
            fto = st + icd + sympy.Integer(offchip_oti[nid])
        else:
            fto = st + icd + t_fire

        # End time: last output tile arrives (N_fire/OTPC - 1) OCI intervals
        # after the first tile. Equivalent to (N_fire - 1) * OTI.
        end = fto + (sympy.ceiling(n_fire / otpc) - 1) * oci

        info[nid] = {
            "node": n,
            "T_fire": t_fire,
            "N_fire": n_fire,
            "OTPC": otpc,
            "ICI": ici,
            "OCI": oci,
            "OTI": oti,
            "ICD": icd,
            "st": st,
            "fto": fto,
            "end": end,
        }
    
    for nid in info:
        for attr in info[nid]:
            if hasattr(info[nid][attr], 'free_symbols'):
                info[nid][attr] = info[nid][attr].xreplace(sym_subs)

    return info


def _compute_contention_oti(offchip_nodes, offchip_requests, offchip_pardispatch,
                             hw_config, pass_info):
    """Compute per-op HBM OTI based on interval overlap from a timing pass."""
    offchip_intervals = {}
    for n in offchip_nodes:
        nid = n.instance_id
        st_val = int(sympy.N(pass_info[nid]["st"]))
        end_val = max(st_val + 1, int(sympy.N(pass_info[nid]["end"])))
        offchip_intervals[nid] = (st_val, end_val)

    offchip_oti = {}
    for n in offchip_nodes:
        nid = n.instance_id
        my_st, my_end = offchip_intervals[nid]
        my_dur = my_end - my_st
        R = offchip_requests[nid]
        P = offchip_pardispatch[nid]

        # Rate-weighted contention: each concurrent op contributes requests
        # proportional to overlap AND its firing rate relative to this op.
        # An op with R=96 but OCI=992 sends 96/992 ≈ 0.1 req/cycle — it
        # shouldn't inflate contention like an op sending 96 req/tile at
        # OCI=96 would. Scale by min(1, my_OCI / other_OCI).
        my_oci = max(1, int(sympy.N(pass_info[nid]["OCI"])))

        concurrent_R = 0
        for other in offchip_nodes:
            oid = other.instance_id
            o_st, o_end = offchip_intervals[oid]
            overlap_st = max(my_st, o_st)
            overlap_end = min(my_end, o_end)
            if overlap_st < overlap_end:
                overlap_frac = (overlap_end - overlap_st) / my_dur
                # Scale by relative firing rate: a slow op (high OCI)
                # injects fewer requests per target tile access.
                o_oci = max(1, int(sympy.N(pass_info[oid]["OCI"])))
                rate_scale = min(1.0, my_oci / o_oci)
                concurrent_R += offchip_requests[oid] * overlap_frac * rate_scale

        concurrent_R = max(R, int(math.ceil(concurrent_R)))
        offchip_oti[nid] = _compute_hbm_oti(R, P, hw_config, concurrent_R)

    return offchip_oti


def _compute_dyndim_expected_values(nodes):
    """Compute expected-value substitutions for symbolic dimensions.

    FlatPartition creates DynDim symbols for per-consumer tile counts.
    FlatReassemble creates DynDim symbols for reassembled elements per
    control firing. Both are resolved assuming uniform routing.

    Must run before timing passes so DynDims are substituted early in
    rate computations (N_fire, NIT, OTPC).
    """
    sym_expected = {}

    # FlatPartition: expected tiles per consumer = input_N_fire / num_consumers
    for n in nodes:
        if isinstance(n, FlatPartition) and n.num_consumers > 1:
            inp_node = n.input if isinstance(n.input, StepOps) else n.input[0]
            inp_nfire = inp_node.N_fire()
            if not hasattr(inp_nfire, 'free_symbols') or not inp_nfire.free_symbols:
                expected_per_consumer = max(1, int(inp_nfire) // n.num_consumers)
                for stream in n.stream_list:
                    for dim in stream.shape:
                        if hasattr(dim, 'expr'):
                            sym_expected[dim.expr] = expected_per_consumer

    # FlatReassemble: expected elements per control = sum(input_N_fires) / control_N_fire
    # Input N_fires may contain FlatPartition symbols, so substitute those first.
    for n in nodes:
        if isinstance(n, FlatReassemble):
            ctrl_node = n.control if isinstance(n.control, StepOps) else n.control[0]
            ctrl_nfire = ctrl_node.N_fire()
            if sym_expected:
                ctrl_nfire = ctrl_nfire.subs(sym_expected)
            total_input_nfire = sympy.Integer(0)
            for inp in n._inputs:
                inp_node = inp if isinstance(inp, StepOps) else inp[0]
                inp_nfire = inp_node.N_fire()
                if sym_expected:
                    inp_nfire = inp_nfire.subs(sym_expected)
                total_input_nfire += inp_nfire
            assert not total_input_nfire.free_symbols, \
                f"FlatReassemble {n} has unresolved symbols in input N_fires: {total_input_nfire.free_symbols}"
            assert not ctrl_nfire.free_symbols, \
                f"FlatReassemble {n} has unresolved symbols in control N_fire: {ctrl_nfire.free_symbols}"
            expected_per_ctrl = max(1, int(total_input_nfire) // max(1, int(ctrl_nfire)))
            for dim in n.stream.shape:
                if hasattr(dim, 'expr'):
                    sym_expected[dim.expr] = expected_per_ctrl

    return sym_expected


def analyze_timing(graph: MultiDiGraph, hw_config: dict = None) -> dict:
    """Run the analytical timing model on a STeP graph.

    Two-pass approach for HBM contention:
      Pass 1: No contention to estimate op time intervals
      Pass 2: Compute per-op contention from actually-concurrent ops
    """
    if hw_config is None:
        hw_config = DEFAULT_HW_CONFIG.copy()

    nodes = list(topological_sort(graph))
    for n in nodes:
        assert isinstance(n, StepOps), f"Non-StepOps node in graph: {n}"

    hbm_addr_offset = hw_config.get("hbm_addr_offset", 64)

    offchip_nodes = [n for n in nodes if n.is_offchip_memory_op()]

    # Per-op HBM info
    offchip_requests = {}
    offchip_pardispatch = {}
    for n in offchip_nodes:
        nid = n.instance_id
        tile_bytes = _get_tile_bytes(n)
        offchip_requests[nid] = math.ceil(tile_bytes / hbm_addr_offset)
        offchip_pardispatch[nid] = _get_par_dispatch(n)

    sym_expected = _compute_dyndim_expected_values(nodes)

    # --- Pass 1: no contention (each op alone) ---
    offchip_oti_p1 = {}
    for n in offchip_nodes:
        nid = n.instance_id
        R = offchip_requests[nid]
        P = offchip_pardispatch[nid]
        offchip_oti_p1[nid] = _compute_hbm_oti(R, P, hw_config, R)

    pass1 = _run_timing_pass(nodes, graph, hw_config, offchip_oti_p1,
                             sym_subs=sym_expected)

    offchip_oti_p2 = _compute_contention_oti(
        offchip_nodes, offchip_requests, offchip_pardispatch,
        hw_config, pass1)

    # --- Pass 2: with contention + buffer absorption ---
    info = _run_timing_pass(nodes, graph, hw_config, offchip_oti_p2,
                            adjust_reshape_padding=True,
                            sym_subs=sym_expected)

    # Total = max end across leaf nodes
    leaf_nodes = [n for n in nodes if graph.out_degree(n) == 0]
    assert len(leaf_nodes) > 0, "Graph has no leaf nodes"
    total_cycles = sympy.Integer(0)
    for leaf in leaf_nodes:
        total_cycles = sympy.Max(total_cycles, info[leaf.instance_id]["end"])

    return {
        "total_cycles": total_cycles,
        "per_node": info,
        "sym_subs": sym_expected,
    }
