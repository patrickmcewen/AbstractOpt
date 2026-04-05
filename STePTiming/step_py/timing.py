"""Analytical timing model for STeP dataflow graphs.

Implements the rate-based performance model from Step_timing.pdf.
Walks the graph in topological order computing per-node timing quantities,
then returns the total graph execution time in cycles.
"""

import math
import sympy
from networkx import MultiDiGraph, topological_sort
from step_py.ops import StepOps, get_stream
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
    from step_py.ops import Bufferize, Streamify
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
    """
    pmu_read = 0
    try:
        inputs = node.input_list
    except (NotImplementedError, AttributeError):
        return 0

    preds = list(graph.predecessors(node))

    for inp in inputs:
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


def _is_leq(a, b, sym_subs=None):
    """Check if sympy expression a <= b, conservatively returning False if uncertain."""
    diff = b - a
    if sym_subs and diff.free_symbols:
        diff = diff.subs(sym_subs)
    diff = sympy.simplify(diff)
    if diff.free_symbols:
        return False
    return int(sympy.N(diff)) >= 0



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

    # Backpressure from bounded DAM channels (depth=2) between operators.
    # Per-tile traces show ~3 cycles additive overhead in steady state.
    backpressure_overhead = 3

    return max(1, bottleneck + startup_overhead + backpressure_overhead)




def _run_timing_pass(nodes, graph, hw_config, offchip_oti,
                     adjust_reshape_padding=False, buffer_absorption=False,
                     sym_subs={}):
    """Run one pass of the timing model with given off-chip OTI values."""
    info = {}

    for n in nodes:
        nid = n.instance_id
        preds = list(graph.predecessors(n))

        n_fire = n.N_fire()
        otpc = n.OTPC()
        nit_map = n.NIT()

        from step_py.ops import Streamify as _Streamify
        if n.is_offchip_memory_op():
            t_fire = sympy.Integer(0)
        elif getattr(n, 'write_back_mu', None) is None and not hasattr(n, 'compute_bw'):
            # Pass-through op (Broadcast, RepeatStatic, etc.)
            # These don't do compute or PMU reads — they forward tile references.
            # T_fire = 0 in the sim (no trace events emitted for Broadcast).
            t_fire = sympy.Integer(0)
        else:
            # T_fire from the op = max(comp_cycles, store_cycles)
            t_fire_base = n.T_fire(hw_config)
            # Add PMU read cost: sim does max(load_cycles, comp_cycles, store_cycles)
            # Only charged for inputs from PMU-backed sources (off-chip loads, Bufferize, etc.)
            pmu_read_cycles = _get_pmu_read_cycles(n, graph)
            t_fire = sympy.Max(t_fire_base, sympy.Integer(pmu_read_cycles))

        # FlatReassemble fan-in: collects tiles from N parallel expert paths.
        # Expert outputs arrive interleaved (round-robin), so the effective
        # per-predecessor OTI is divided by the number of data inputs.
        from step_py.ops import FlatReassemble
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
        absorbed_preds = []
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
                """# PMU buffer absorption: if predecessor stores to PMU and has
                # finished before this node starts, skip it from OCI — tiles
                # are all in PMU, consumer reads at its own rate.
                if buffer_absorption and _produces_pmu_tile(pred):
                    pred_end = info[pid]["end"]
                    if _is_leq(pred_end, st, sym_subs):
                        print(f"absorbing pred {pred.instance_id} because pred_end <= st")
                        absorbed_preds.append(pred)
                        continue  # Skip this pred — handled by end constraint"""
                nit_count = nit_map.get(pid, sympy.Integer(1))
                oci = sympy.Max(oci, nit_count * sympy.Max(t_fire, pred_oti))
            oci = sympy.Max(t_fire, oci)

        oti = oci / otpc

        # Reshape with padding: padding tiles are generated locally at zero
        # cost (no waiting for predecessor). Within each chunk of chunk_size
        # tiles, some may be padding, so the effective per-tile OTI is lower.
        # Approximation: OTI = OCI / chunk_size (uniform padding distribution).
        from step_py.ops import Reshape
        if adjust_reshape_padding and isinstance(n, Reshape) and n.pad_fn is not None:
            oti = oci / sympy.Integer(n.chunk_size)

        # FlatPartition fan-out: tiles are distributed round-robin to N consumers.
        # Each consumer sees tiles at N times the interval.
        from step_py.ops import FlatPartition
        if isinstance(n, FlatPartition):
            oti = oti * sympy.Integer(n.num_consumers)

        # ICD = max over preds of ((NIT - 1) * OTI(pred))
        # Absorbed PMU preds are skipped — no initial collection delay from PMU.
        icd = sympy.Integer(0)
        for pred in preds:
            if pred in absorbed_preds:
                continue  # Already absorbed
            pid = pred.instance_id
            pred_oti = info[pid]["OTI"] / fan_in_divisor
            nit_count = nit_map.get(pid, sympy.Integer(1))
            icd = sympy.Max(icd, (nit_count - 1) * sympy.Max(t_fire, pred_oti))

        # First tile out: off-chip ops use their OTI (which already includes startup)
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
                             hw_config, pass_info, has_symbolic, sym_subs):
    """Compute per-op HBM OTI based on interval overlap from a timing pass."""
    offchip_intervals = {}
    for n in offchip_nodes:
        nid = n.instance_id
        st_expr = pass_info[nid]["st"]
        end_expr = pass_info[nid]["end"]
        if has_symbolic and sym_subs:
            st_expr = st_expr.subs(sym_subs)
            end_expr = end_expr.subs(sym_subs)
        st_val = int(sympy.N(st_expr))
        end_val = max(st_val + 1, int(sympy.N(end_expr)))
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
        my_oci_expr = pass_info[nid]["OCI"]
        if has_symbolic and sym_subs:
            my_oci_expr = my_oci_expr.subs(sym_subs)
        my_oci = max(1, int(sympy.N(my_oci_expr)))

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
                o_oci_expr = pass_info[oid]["OCI"]
                if has_symbolic and sym_subs:
                    o_oci_expr = o_oci_expr.subs(sym_subs)
                o_oci = max(1, int(sympy.N(o_oci_expr)))
                rate_scale = min(1.0, my_oci / o_oci)
                concurrent_R += offchip_requests[oid] * overlap_frac * rate_scale

        concurrent_R = max(R, int(math.ceil(concurrent_R)))
        offchip_oti[nid] = _compute_hbm_oti(R, P, hw_config, concurrent_R)

    return offchip_oti


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

    # --- Compute expected-value substitutions for symbolic dimensions ---
    # Derive expected per-consumer values from FlatPartition nodes.
    # Each FlatPartition creates SelectGen symbols for its consumers.
    # Expected value = input_N_fire / num_consumers (uniform routing).
    # Computed before pass 1 so DynDims are substituted early in rate
    # computations (N_fire, NIT, OTPC) rather than deferred to the end.
    from step_py.ops import FlatPartition, FlatReassemble
    sym_expected = {}
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
    
    # Derive expected values for FlatReassemble dynamic dimensions.
    # FlatReassemble gathers from multiple inputs into one stream; its DynDim
    # represents reassembled elements per control firing.
    # Expected value = sum(input_N_fires) / control_N_fire (uniform routing).
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

    # --- Pass 1: no contention (each op alone) ---
    offchip_oti_p1 = {}
    for n in offchip_nodes:
        nid = n.instance_id
        R = offchip_requests[nid]
        P = offchip_pardispatch[nid]
        offchip_oti_p1[nid] = _compute_hbm_oti(R, P, hw_config, R)

    pass1 = _run_timing_pass(nodes, graph, hw_config, offchip_oti_p1,
                             sym_subs=sym_expected)

    # Always seed sym_subs from sym_expected so both passes get DynDim values.
    # Pass 1 early substitution makes expressions concrete, so we can't rely
    # on collecting free_symbols from pass 1 results to build sym_subs.
    sym_subs = dict(sym_expected)

    # Offchip-scoped contention estimation (changing scope alters which
    # intervals overlap, breaking contention accuracy).
    has_symbolic = any(
        pass1[n.instance_id]["end"].free_symbols for n in offchip_nodes
    )
    if has_symbolic:
        all_symbols = set()
        for n in offchip_nodes:
            nid = n.instance_id
            all_symbols |= pass1[nid]["st"].free_symbols
            all_symbols |= pass1[nid]["end"].free_symbols

        for s in all_symbols:
            if s not in sym_subs:
                raise ValueError(f"Symbol {s} not found in sym_subs")

    offchip_oti_p2 = _compute_contention_oti(
        offchip_nodes, offchip_requests, offchip_pardispatch,
        hw_config, pass1, has_symbolic, sym_subs)

    # --- Pass 2: with contention + buffer absorption ---
    info = _run_timing_pass(nodes, graph, hw_config, offchip_oti_p2,
                            adjust_reshape_padding=True, buffer_absorption=True,
                            sym_subs=sym_subs)

    # Total = max end across leaf nodes
    leaf_nodes = [n for n in nodes if graph.out_degree(n) == 0]
    assert len(leaf_nodes) > 0, "Graph has no leaf nodes"
    total_cycles = sympy.Integer(0)
    for leaf in leaf_nodes:
        total_cycles = sympy.Max(total_cycles, info[leaf.instance_id]["end"])

    return {
        "total_cycles": total_cycles,
        "per_node": info,
        "sym_subs": sym_subs,
    }
