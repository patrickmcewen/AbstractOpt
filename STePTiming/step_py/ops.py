from abc import ABC, abstractmethod
from sympy import ceiling, symbols, Piecewise, Integer
import torch
from typing import List, Optional, Tuple, Union
from step_py.dyndim import DynDim
from step_py.functions.accum_fn import AccumFn
from step_py.functions.init_fn import InitFn
from step_py.functions.map_accum_fn import MapAccumFn
from step_py.functions.map_fn import MapFn
from step_py.datatype import (
    Bool,
    Buffer,
    DynTile,
    MultiHot,
    Stream,
    Tile,
    Select,
    Float16,
    Float32,
    Uint64,
)
from networkx import MultiDiGraph
import sympy


def get_stream(input: Union["StepOps", Tuple["StepOps", int]]) -> Stream:
    if isinstance(input, StepOps):
        return input.stream

    if (
        isinstance(input, Tuple)
        and isinstance(input[0], StepOps)
        and isinstance(input[1], int)
    ):
        input_node, stream_idx = input
        return input_node.stream_idx(stream_idx)
    else:
        raise TypeError(f"Wrong input type! Input: {input} (type:{type(input)})")


PMU_BW = 64  # on-chip scratchpad bandwidth in bytes/cycle (matches simulator)


def _compute_flops_per_tile(fn, input_streams: list, output_stream) -> sympy.Expr:
    """Estimate FLOPs for one firing of a Map/Accum function."""
    from step_py.functions import map_fn, map_accum_fn

    out_dtype = output_stream.stream_dtype
    # Matmul: 2*M*K*N
    if isinstance(fn, (map_fn.Matmul, map_fn.DynMatmul, map_accum_fn.Matmul, map_accum_fn.DynMatmul)):
        tile_a = input_streams[0].stream_dtype
        tile_b = input_streams[1].stream_dtype
        m, k = tile_a.shape[0], tile_a.shape[1]
        n = tile_b.shape[1] if not fn.weight_transposed else tile_b.shape[0]
        return sympy.Integer(2) * m * k * n

    # RowWiseSum: tile_row * tile_col ops
    if isinstance(fn, map_fn.RowWiseSum):
        in_tile = input_streams[0].stream_dtype
        return sympy.Integer(in_tile.shape[0] * in_tile.shape[1])

    # Default: 1 op per output element
    if isinstance(out_dtype, (Tile, DynTile)):
        r, c = out_dtype.shape[0], out_dtype.shape[1]
        if isinstance(r, int) and isinstance(c, int):
            return sympy.Integer(r * c)
        return r * c
    return sympy.Integer(1)


def _compute_roofline(flops, compute_bw, output_tile_bytes, write_back_mu) -> sympy.Expr:
    """Roofline T_fire: max(comp_cycles, store_to_pmu_cycles).
    compute_bw is in FLOPs/cycle. PMU_BW is 64 bytes/cycle."""
    comp_cycles = sympy.ceiling(flops / compute_bw)
    store_cycles = sympy.ceiling(output_tile_bytes / PMU_BW) if write_back_mu else sympy.Integer(0)
    return sympy.Max(comp_cycles, store_cycles)


class StepOps(ABC):
    _counter: int = 0
    instance_id: int

    def __init__(self):
        self.instance_id = StepOps._counter
        StepOps._counter += 1

    @property
    @abstractmethod
    def stream(self) -> Stream:
        """The stream of the operation."""
        pass

    @property
    @abstractmethod
    def stream_list(self) -> List[Stream]:
        pass

    @property
    @abstractmethod
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        pass

    @property
    @abstractmethod
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        pass

    @abstractmethod
    def stream_idx(self, idx: int) -> Stream:
        """The stream of the operation."""
        pass

    @abstractmethod
    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        pass

    @abstractmethod
    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        pass

    @abstractmethod
    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic (bytes) for this operation."""
        pass

    def is_offchip_memory_op(self) -> bool:
        """Override to True in off-chip memory operators."""
        return False

    @abstractmethod
    def T_fire(self, hw_config: dict) -> sympy.Expr:
        """Firing time: processing time per firing (Roofline). Returns cycles."""
        pass

    @abstractmethod
    def N_fire(self) -> sympy.Expr:
        """Firing count: total number of firings."""
        pass

    @abstractmethod
    def NIT(self) -> dict:
        """Input tile count per predecessor per firing.
        Returns {predecessor_instance_id: sympy.Expr}.
        For source nodes, returns empty dict."""
        pass

    @abstractmethod
    def OTPC(self) -> sympy.Expr:
        """Output tiles per chunk: output tiles produced per firing."""
        pass


class MockStreamOp(StepOps):
    _stream: Stream
    _aten_op_name: str
    _output_idx: int

    def __init__(self, stream: Stream, aten_op_name: str = "", output_idx: int = 0):
        super().__init__()
        self._stream = stream
        self._aten_op_name = aten_op_name
        self._output_idx = output_idx

    @property
    def stream(self) -> Stream:
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError("MockStreamOp doesn't have input")

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        raise NotImplementedError("MockStreamOp doesn't have input")

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        raise NotImplementedError("MockStreamOp doesn't have input")

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        raise NotImplementedError(
            "MockStreamOp shouldn't remain in the graph when calling this method."
        )

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic (bytes) for this operation."""
        raise NotImplementedError(
            "MockStreamOp shouldn't remain in the graph when calling this method."
        )

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return sympy.Integer(0)

    def NIT(self) -> dict:
        return {}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)


class RandomOffChipLoad(StepOps):
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    raddr: Union[StepOps, Tuple[StepOps, int]]
    tile_row: int
    tile_col: int
    n_byte: int
    base_addr_byte: int
    par_dispatch: int
    _stream: Stream
    transposed: bool
    track_traffic: bool

    def __init__(
        self,
        graph: MultiDiGraph,
        underlying: torch.Tensor,
        raddr: Union[StepOps, Tuple[StepOps, int]],
        tile_row: int,
        tile_col: int,
        base_addr_byte: int,
        par_dispatch: int,
        mock_bf16: bool = False,
        transposed: bool = False,
    ):
        super().__init__()

        self.underlying = underlying
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.raddr = raddr
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.base_addr_byte = base_addr_byte
        self.par_dispatch = par_dispatch
        self.transposed = transposed

        if underlying.dtype == torch.float32:
            if mock_bf16:
                self.n_byte = 2  # we will use this to mimic bfloat16
            else:
                self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=get_stream(raddr).shape
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=get_stream(raddr).shape
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

        raddr_node = raddr if isinstance(raddr, StepOps) else raddr[0]
        graph.add_edge(raddr_node, self)

        self.track_traffic = False

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.raddr

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.raddr]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union[StepOps, Tuple[StepOps, int]],
        new_input: Union[StepOps, Tuple[StepOps, int]],
    ):
        if get_stream(self.raddr) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self.raddr = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(self.tile_row * self.tile_col * self.n_byte)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        raddr_node = self.raddr if isinstance(self.raddr, StepOps) else self.raddr[0]
        return {raddr_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True


class RandomOffChipStore(StepOps):
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    wdata: Union[StepOps, Tuple[StepOps, int]]
    waddr: Union[StepOps, Tuple[StepOps, int]]
    tile_row: int
    tile_col: int
    n_byte: int
    store_file_name: str
    ack_based_on_waddr: bool  # if true, the ack stream's shape will be based on the waddr, otherwise it is based on the wdata
    base_addr_byte: int
    buffer_depth: int
    par_dispatch: int
    _stream: Stream
    has_done_stream: bool

    def __init__(
        self,
        graph: MultiDiGraph,
        underlying: torch.Tensor,
        wdata: Union[StepOps, Tuple[StepOps, int]],
        waddr: Union[StepOps, Tuple[StepOps, int]],
        tile_row: int,
        tile_col: int,
        base_addr_byte: int,
        par_dispatch: int,
        ack_based_on_waddr: bool = True,
        buffer_depth: int = 1,
        mock_bf16: bool = False,
        has_done_stream: bool = False,
    ):
        super().__init__()

        self.underlying = underlying
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.wdata = wdata
        self.waddr = waddr
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.base_addr_byte = base_addr_byte
        self.par_dispatch = par_dispatch
        self.buffer_depth = buffer_depth
        self.ack_based_on_waddr = ack_based_on_waddr
        self.store_file_name = f"{str(self)}.npy"
        self.has_done_stream = has_done_stream

        if underlying.dtype == torch.float32:
            if mock_bf16:
                self.n_byte = 2  # we will use this to mimic bfloat16
            else:
                self.n_byte = 4

            self._stream = Stream(
                stream_dtype=Bool(),
                shape=(
                    get_stream(waddr).shape
                    if ack_based_on_waddr
                    else get_stream(wdata).shape
                ),
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            self._stream = Stream(
                stream_dtype=Bool(),
                shape=(
                    get_stream(waddr).shape
                    if ack_based_on_waddr
                    else get_stream(wdata).shape
                ),
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

        waddr_node = waddr if isinstance(waddr, StepOps) else waddr[0]
        wdata_node = wdata if isinstance(wdata, StepOps) else wdata[0]
        graph.add_edge(waddr_node, self)
        graph.add_edge(wdata_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return NotImplementedError(
            "Shouldn't be called for nodes that has multiple inputs"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.waddr, self.wdata]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union[StepOps, Tuple[StepOps, int]],
        new_input: Union[StepOps, Tuple[StepOps, int]],
    ):
        if self.waddr == org_input:
            if get_stream(self.waddr) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.waddr = new_input
        elif self.wdata == org_input:
            if get_stream(self.wdata) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.wdata = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte * self.buffer_depth
        )

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        waddr_node = self.waddr if isinstance(self.waddr, StepOps) else self.waddr[0]
        wdata_node = self.wdata if isinstance(self.wdata, StepOps) else self.wdata[0]
        return {waddr_node.instance_id: sympy.Integer(1), wdata_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True


class DynLinearOffChipLoad(StepOps):
    input_tensor_name: str
    tensor_shape_tiled: Tuple[Union[int, DynDim], ...]
    tile_row: int
    tile_col: int
    n_byte: int
    par_dispatch: int
    _stream: Stream

    def __init__(
        self,
        input_tensor_name: str,
        tensor_shape_tiled: Tuple[Union[int, DynDim], ...],
        dtype: torch.dtype,
        tile_row: int,
        tile_col: int,
        par_dispatch: int,
        mock_bf16: bool = False,
    ):
        super().__init__()

        self.input_tensor_name = input_tensor_name
        self.tensor_shape_tiled = tensor_shape_tiled

        self.tile_row = tile_row
        self.tile_col = tile_col
        self.par_dispatch = par_dispatch

        if dtype == torch.float32:
            if mock_bf16:
                self.n_byte = 2  # we will use this to mimic bfloat16
            else:
                self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.tensor_shape_tiled
            )
        elif dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.tensor_shape_tiled
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union[StepOps, Tuple[StepOps, int]],
        new_input: Union[StepOps, Tuple[StepOps, int]],
    ):
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(self.tile_row * self.tile_col * self.n_byte)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        return {}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True



class LinearOffChipLoad(StepOps):
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    stride: Tuple[int, ...]
    out_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    n_byte: int
    par_dispatch: int
    _stream: Stream
    save_npy: bool
    transposed: bool

    def __init__(
        self,
        underlying: torch.Tensor,
        stride: Tuple[int, ...],
        out_shape_tiled: Tuple[int, ...],
        tile_row: int,
        tile_col: int,
        par_dispatch: int,
        mock_bf16: bool = False,
        save_npy: bool = False,
        transposed: bool = False,
    ):
        super().__init__()

        self.underlying = underlying
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.stride = stride
        self.out_shape_tiled = out_shape_tiled
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.par_dispatch = par_dispatch
        self.save_npy = save_npy
        self.transposed = transposed

        if underlying.dtype == torch.float32:
            if mock_bf16:
                self.n_byte = 2  # we will use this to mimic bfloat16
            else:
                self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.out_shape_tiled
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.out_shape_tiled
            )
        elif underlying.dtype == torch.uint64:
            self.n_byte = 8

            stream_dtype = Tile(
                tile_dtype=Uint64(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.out_shape_tiled
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union[StepOps, Tuple[StepOps, int]],
        new_input: Union[StepOps, Tuple[StepOps, int]],
    ):
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(self.tile_row * self.tile_col * self.n_byte)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        return {}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True



class LinearOffChipLoadRef(StepOps):
    ref: Union[StepOps, Tuple[StepOps, int]]
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    stride: Tuple[int, ...]
    out_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    n_byte: int
    par_dispatch: int
    _stream: Stream
    transposed: bool
    trigger_rank: (
        int  # number of the inner dimensions that are used to trigger the load
    )

    def __init__(
        self,
        graph: MultiDiGraph,
        ref: Union[StepOps, Tuple[StepOps, int]],
        underlying: torch.Tensor,
        stride: Tuple[int, ...],
        out_shape_tiled: Tuple[int, ...],
        tile_row: int,
        tile_col: int,
        par_dispatch: int,
        mock_bf16: bool = False,
        transposed: bool = False,
        trigger_rank: int = 0,
    ):
        super().__init__()

        self.ref = ref
        self.underlying = underlying
        self.trigger_rank = trigger_rank
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.stride = stride
        self.out_shape_tiled = out_shape_tiled
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.par_dispatch = par_dispatch
        self.transposed = transposed

        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(ref_node, self)

        ref_stream: Stream = get_stream(ref)
        ref_stream_shape = (
            ref_stream.shape[: -self.trigger_rank]
            if self.trigger_rank > 0
            else ref_stream.shape
        )
        if underlying.dtype == torch.float32:
            if mock_bf16:
                self.n_byte = 2  # we will use this to mimic bfloat16
            else:
                self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=ref_stream_shape + self.out_shape_tiled
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col) if not transposed else (tile_col, tile_row),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=ref_stream_shape + self.out_shape_tiled
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.ref

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} (rank={self.trigger_rank})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.ref) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self.ref = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        # print(f"LinearOffChipLoadRef: total_elements: {total_elements}")
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(self.tile_row * self.tile_col * self.n_byte)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        ref_node = self.ref if isinstance(self.ref, StepOps) else self.ref[0]
        return {ref_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True



class ExpandRef(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    ref: Union[StepOps, Tuple[StepOps, int]]
    expand_rank: int  # number of the inner dimensions with size 1 that will be expanded
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        ref: Union[StepOps, Tuple[StepOps, int]],
        expand_rank: int,
    ):
        super().__init__()

        in_stream_shape = get_stream(input).shape
        ref_stream_shape = get_stream(ref).shape

        assert expand_rank > 0
        assert in_stream_shape[-expand_rank:] == (1,) * expand_rank
        assert in_stream_shape[:-expand_rank] == ref_stream_shape[:-expand_rank]

        self._input = input
        self.ref = ref
        self._stream = Stream(
            stream_dtype=get_stream(input).stream_dtype, shape=get_stream(ref).shape
        )
        self.expand_rank = expand_rank

        input_node = input if isinstance(input, StepOps) else input[0]
        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(input_node, self)
        graph.add_edge(ref_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple inputs"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.ref == org_input:
            if get_stream(self.ref) != get_stream(new_input):
                raise ValueError("The shape of the ref stream shouldn't change")
            self.ref = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        stream_size: sympy.Expr = self._stream.stream_dtype.size_in_bytes()

        if count_fifos:
            return sympy.Mul(stream_size, sympy.Integer(2))
        else:
            return stream_size

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        ref_node = self.ref if isinstance(self.ref, StepOps) else self.ref[0]
        return {input_node.instance_id: sympy.Integer(1), ref_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class RepeatRef(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    ref: Union[StepOps, Tuple[StepOps, int]]
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        ref: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()

        in_stream_shape = get_stream(input).shape
        ref_stream_shape = get_stream(ref).shape

        assert in_stream_shape == ref_stream_shape[:-1]

        self._input = input
        self.ref = ref
        self._stream = Stream(
            stream_dtype=get_stream(input).stream_dtype, shape=get_stream(ref).shape
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(input_node, self)
        graph.add_edge(ref_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple inputs"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.ref == org_input:
            if get_stream(self.ref) != get_stream(new_input):
                raise ValueError("The shape of the ref stream shouldn't change")
            self.ref = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        stream_size: sympy.Expr = self._stream.stream_dtype.size_in_bytes()

        if count_fifos:
            return sympy.Mul(stream_size, sympy.Integer(2))
        else:
            return stream_size

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        ref_node = self.ref if isinstance(self.ref, StepOps) else self.ref[0]
        return {input_node.instance_id: sympy.Integer(1), ref_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class RepeatStatic(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    repeat_factor: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        repeat_factor: int,
    ):
        super().__init__()
        self._input = input
        self.repeat_factor = repeat_factor

        input_stream: Stream = get_stream(input)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(input_stream.shape + (repeat_factor,)),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        stream_size: sympy.Expr = self._stream.stream_dtype.size_in_bytes()

        if count_fifos:
            return sympy.Mul(stream_size, sympy.Integer(2))
        else:
            return stream_size

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Promote(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    promote_rank: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        promote_rank: int,
    ):
        super().__init__()
        self._input = input
        self.promote_rank = promote_rank

        input_stream: Stream = get_stream(input)
        stream_shape = list(input_stream.shape)
        stream_shape.insert(len(input_stream.shape) - promote_rank, 1)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(stream_shape),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}({self.promote_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the size of the stream's data type
        stream_size: sympy.Expr = self._stream.stream_dtype.size_in_bytes()
        return sympy.Mul(stream_size, sympy.Integer(2))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class PromoteOuter(StepOps):
    """
    [Dn,...,D0] => [1,Dn,...,D0]
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()
        self._input = input

        input_stream: Stream = get_stream(input)
        outer_most_dim = input_stream.shape[0]

        new_outer_most_expr = 1
        if isinstance(outer_most_dim, DynDim):
            outer_most_expr: sympy.Expr = outer_most_dim.expr
            new_outer_most_expr = DynDim(
                Piecewise(
                    (Integer(1), outer_most_expr >= 1),
                    (Integer(0), outer_most_expr < 1),
                )
            )
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=(new_outer_most_expr,) + input_stream.shape,
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        stream_size: sympy.Expr = self._stream.stream_dtype.size_in_bytes()

        if not count_fifos:
            return stream_size

        return sympy.Mul(stream_size, sympy.Integer(2))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class BinaryMap(StepOps):
    in1: Union[StepOps, Tuple[StepOps, int]]
    in2: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        in1: Union[StepOps, Tuple[StepOps, int]],
        in2: Union[StepOps, Tuple[StepOps, int]],
        fn: MapFn,
        write_back_mu: bool,
        compute_bw: int,
    ):

        super().__init__()

        self.in1 = in1
        self.in2 = in2
        self.fn = fn
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in1_stream: Stream = get_stream(in1)
        in2_stream: Stream = get_stream(in2)

        for dim in range(len(in1_stream.shape)):
            if isinstance(in1_stream.shape[dim], int):
                if in1_stream.shape[dim] != in2_stream.shape[dim]:
                    raise ValueError(
                        f"Input streams must have the same shape for the static dims: {in1_stream.shape} != {in2_stream.shape}"
                    )

        self._stream = Stream(
            stream_dtype=self.fn.apply(
                (in1_stream.stream_dtype, in2_stream.stream_dtype)
            ),
            shape=in1_stream.shape,
        )

        input_node1 = in1 if isinstance(in1, StepOps) else in1[0]
        input_node2 = in2 if isinstance(in2, StepOps) else in2[0]

        graph.add_edge(input_node1, self)
        graph.add_edge(input_node2, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.in1

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.in1, self.in2]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} (fn: {self.fn})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.in1 == org_input:
            if get_stream(self.in1) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in1 = new_input
        elif self.in2 == org_input:
            if get_stream(self.in2) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in2 = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get streams for inputs
        in1_stream = get_stream(self.in1)
        in2_stream = get_stream(self.in2)

        # Calculate tile sizes in bytes for each stream
        assert not isinstance(
            in1_stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        assert not isinstance(
            in2_stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        assert not isinstance(
            self._stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."

        in1_stream_dtype_size = in1_stream.stream_dtype.size_in_bytes()
        in2_stream_dtype_size = in2_stream.stream_dtype.size_in_bytes()
        out_stream_dtype_size = self._stream.stream_dtype.size_in_bytes()

        return sympy.Add(
            in1_stream_dtype_size, in2_stream_dtype_size, out_stream_dtype_size
        )

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        in1_stream = get_stream(self.in1)
        in2_stream = get_stream(self.in2)
        flops = _compute_flops_per_tile(self.fn, [in1_stream, in2_stream], self._stream)
        out_tile_bytes = self._stream.stream_dtype.size_in_bytes()
        return _compute_roofline(flops, self.compute_bw, out_tile_bytes, self.write_back_mu)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        in1_node = self.in1 if isinstance(self.in1, StepOps) else self.in1[0]
        in2_node = self.in2 if isinstance(self.in2, StepOps) else self.in2[0]
        return {in1_node.instance_id: sympy.Integer(1), in2_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class BinaryMapAccum(StepOps):
    in1: Union[StepOps, Tuple[StepOps, int]]
    in2: Union[StepOps, Tuple[StepOps, int]]
    fn: MapAccumFn
    init_fn: InitFn
    rank: int
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    # [Genghan] We need an init function?
    def __init__(
        self,
        graph: MultiDiGraph,
        in1: Union[StepOps, Tuple[StepOps, int]],
        in2: Union[StepOps, Tuple[StepOps, int]],
        fn: MapAccumFn,
        init_fn: InitFn,
        rank: int,
        write_back_mu: bool,
        compute_bw: int,
    ):

        super().__init__()

        self.in1 = in1
        self.in2 = in2
        self.fn = fn
        self.init_fn = init_fn
        self.rank = rank
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in1_stream: Stream = get_stream(in1)
        in2_stream: Stream = get_stream(in2)

        assert rank > 0, "Rank must be greater than 0."
        for in1_dim, in2_dim in zip(in1_stream.shape, in2_stream.shape):
            if isinstance(in1_dim, DynDim) and isinstance(in2_dim, DynDim):
                assert in1_dim.expr.equals(
                    in2_dim.expr
                ), f"Input streams must have the same shape. {in1_dim.expr} \n!=\n{in2_dim.expr}"
            else:
                assert (
                    in1_dim == in2_dim
                ), f"Input streams must have the same shape. {in1_dim} \n!=\n{in2_dim}"

        self._stream = Stream(
            stream_dtype=self.fn.apply(
                (in1_stream.stream_dtype, in2_stream.stream_dtype)
            ),
            shape=in1_stream.shape[: -self.rank],
        )

        input_node1 = in1 if isinstance(in1, StepOps) else in1[0]
        input_node2 = in2 if isinstance(in2, StepOps) else in2[0]

        graph.add_edge(input_node1, self)
        graph.add_edge(input_node2, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.in1

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.in1, self.in2]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} (fn: {self.fn})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.in1 == org_input:
            if get_stream(self.in1) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in1 = new_input
        elif self.in2 == org_input:
            if get_stream(self.in2) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in2 = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            # Get the size of the stream's data type
            return self._stream.stream_dtype.size_in_bytes()

        # Get the first input stream (all inputs should have the same datatype)
        in_stream = get_stream(self.in1)

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()

        # Return the size times (len(self._inputs) + 1) for FIFO requirements
        return sympy.Mul(in_tile_size, sympy.Integer(len(self.input_list) + 1))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        in1_stream = get_stream(self.in1)
        in2_stream = get_stream(self.in2)
        flops = _compute_flops_per_tile(self.fn, [in1_stream, in2_stream], self._stream)
        out_tile_bytes = self._stream.stream_dtype.size_in_bytes()
        return _compute_roofline(flops, self.compute_bw, out_tile_bytes, self.write_back_mu)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        in1_stream = get_stream(self.in1)
        k = sympy.Integer(1)
        for dim in in1_stream.shape[-self.rank:]:
            k = k * (dim.expr if isinstance(dim, DynDim) else dim)
        in1_node = self.in1 if isinstance(self.in1, StepOps) else self.in1[0]
        in2_node = self.in2 if isinstance(self.in2, StepOps) else self.in2[0]
        return {in1_node.instance_id: k, in2_node.instance_id: k}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Broadcast(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    num_consumers: int
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        num_consumers: int,
    ):
        super().__init__()

        self._input = input
        self.num_consumers = num_consumers

        in_stream: Stream = get_stream(input)
        self._stream = [
            Stream(stream_dtype=in_stream.stream_dtype, shape=in_stream.shape)
            for _ in range(num_consumers)
        ]

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        return self._stream[idx]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream[0].total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class OffChipStore(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    tensor_shape_tiled: Tuple[Union[int, DynDim], ...]
    tile_row: int
    tile_col: int
    store_file_name: str  # This should not include the file extension!!
    par_dispatch: int

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        par_dispatch: int,
        store_file_name: str = "output",
    ):
        super().__init__()

        self._input = input
        in_stream: Stream = get_stream(input)
        self.tensor_shape_tiled = in_stream.shape[1:]
        assert isinstance(in_stream.stream_dtype, Tile), "Tensor shape must be a tuple."
        self.tile_row = in_stream.stream_dtype.shape[0]
        self.tile_col = in_stream.stream_dtype.shape[1]
        self.store_file_name = store_file_name
        self.par_dispatch = par_dispatch

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def stream_list(self) -> List[Stream]:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes without an output stream"
        )

    def get_untiled_shape(self) -> Tuple[Union[int, DynDim], ...]:
        """Get the un-tiled shape of the tensor."""
        if len(self.tensor_shape_tiled) == 1:
            return (self.tensor_shape_tiled[-1] * self.tile_row, self.tile_col)
        else:
            return self.tensor_shape_tiled[:-2] + (
                self.tensor_shape_tiled[-2] * self.tile_row,
                self.tensor_shape_tiled[-1] * self.tile_col,
            )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            print(f"get_stream(self.input): {get_stream(self.input)}")
            print(f"get_stream(new_input): {get_stream(new_input)}")
            raise ValueError(
                f"The shape of the input stream shouldn't change ({get_stream(self.input)} != {get_stream(new_input)})"
            )
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        # Get the input stream
        input_stream = get_stream(self._input)

        # Calculate the total number of elements in the input stream
        total_elements = input_stream.total_elements()

        assert not isinstance(
            input_stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        # Multiply by the size of the data type
        return sympy.Mul(total_elements, input_stream.stream_dtype.size_in_bytes())

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the input stream
        input_stream = get_stream(self._input)

        # Return the size of the data type of the input stream
        return input_stream.stream_dtype.size_in_bytes()

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return get_stream(self._input).total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True



class DynOffChipStore(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    shape_path: str  # Should contain the untiled shape of the tensor
    tile_row: int
    tile_col: int
    store_file_name: str  # This should not include the file extension!!
    par_dispatch: int

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        shape_file_name: str,
        par_dispatch: int,
        store_file_name: str = "output",
    ):
        super().__init__()

        self._input = input
        in_stream: Stream = get_stream(input)
        self.shape_path = f"{shape_file_name}.json"
        assert isinstance(in_stream.stream_dtype, Tile), "Tensor shape must be a tuple."
        self.tile_row = in_stream.stream_dtype.shape[0]
        self.tile_col = in_stream.stream_dtype.shape[1]
        self.store_file_name = store_file_name
        self.par_dispatch = par_dispatch

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def stream_list(self) -> List[Stream]:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes without an output stream"
        )

    def get_untiled_shape(self) -> Tuple[Union[int, DynDim], ...]:
        raise NotImplementedError(
            "DynOffChipStore does not have a get_untiled_shape method as it will be decided in runtime."
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            print(f"get_stream(self.input): {get_stream(self.input)}")
            print(f"get_stream(new_input): {get_stream(new_input)}")
            raise ValueError(
                f"The shape of the input stream shouldn't change ({get_stream(self.input)} != {get_stream(new_input)})"
            )
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        # Get the input stream
        input_stream = get_stream(self._input)

        # Calculate the total number of elements in the input stream
        total_elements = input_stream.total_elements()

        assert not isinstance(
            input_stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        # Multiply by the size of the data type
        return sympy.Mul(total_elements, input_stream.stream_dtype.size_in_bytes())

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the input stream
        input_stream = get_stream(self._input)

        # Return the size of the data type of the input stream
        return input_stream.stream_dtype.size_in_bytes()

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return get_stream(self._input).total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

    def is_offchip_memory_op(self) -> bool:
        return True



class Bufferize(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    rank: int
    off_chip: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        rank: int,
        off_chip: bool = False,
    ):
        super().__init__()

        self._input = input
        self.rank = rank
        self.off_chip = off_chip

        in_stream: Stream = get_stream(input)
        assert rank > 0, "Rank must be greater than 0."
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        buffer_shape = tuple(in_stream.shape[-rank:])
        self._stream = Stream(
            stream_dtype=Buffer(buff_dtype=in_stream.stream_dtype, shape=buffer_shape),
            shape=in_stream.shape[: -self.rank],
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}({self.rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        if not self.off_chip:
            return sympy.Integer(0)

        # Get the input stream
        in_stream = get_stream(self._input)

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."
        tile_size = in_stream.stream_dtype.size_in_bytes()

        # Calculate the total number of elements in the input stream
        total_elements = in_stream.total_elements()

        return sympy.Mul(total_elements, tile_size)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the input stream
        in_stream = get_stream(self._input)
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."
        tile_size = in_stream.stream_dtype.size_in_bytes()

        if (
            self.off_chip
        ):  # If the buffer is off-chip, only the tile size is counted for the staging area
            return tile_size

        # Get the buffer stream
        assert isinstance(
            self._stream.stream_dtype, Buffer
        ), "Stream must be a Buffer type."
        buffer_size = self._stream.stream_dtype.size_in_bytes()

        return sympy.Add(buffer_size, tile_size)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        in_stream = get_stream(self._input)
        k = sympy.Integer(1)
        for dim in in_stream.shape[-self.rank:]:
            k = k * (dim.expr if isinstance(dim, DynDim) else dim)
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: k}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Streamify(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    repeat_factor: List[int]
    rank: int
    off_chip: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        repeat_factor: List[int],
        rank: int,  # The rank of the Buffer
        off_chip: bool = False,
    ):
        super().__init__()

        self._input = input
        self.repeat_factor = repeat_factor
        self.rank = rank
        self.off_chip = off_chip

        in_stream: Stream = get_stream(input)
        assert rank > 0, "Rank must be greater than 0."
        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Buffer type."

        buffer_shape = in_stream.stream_dtype.shape
        buffer_dtype = in_stream.stream_dtype.buff_dtype
        self._stream = Stream(
            stream_dtype=buffer_dtype,
            shape=in_stream.shape + tuple(repeat_factor) + buffer_shape,
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        if not self.off_chip:
            return sympy.Integer(0)

        # Get the input stream
        in_stream = get_stream(self._input)
        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Buffer type."
        buffer_size = self._stream.stream_dtype.size_in_bytes()

        # Calculate the total number of elements in the input stream
        total_input_elements = in_stream.total_elements()

        # Calculate the product of elements in repeat_factor
        repeat_product = sympy.Mul(*self.repeat_factor)

        return sympy.Mul(total_input_elements, repeat_product, buffer_size)

    def on_chip_requirement(self, count_fifos: bool = True) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        assert isinstance(
            self._stream.stream_dtype, Tile
        ), "Stream must be a Tile type."
        tile_size = self._stream.stream_dtype.size_in_bytes()

        if self.off_chip:
            return tile_size

        in_stream = get_stream(self._input)
        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Stream must be a Buffer type."
        # Calculate the size of the buffer (self._stream's datatype)
        buffer_size = in_stream.stream_dtype.size_in_bytes()

        return sympy.Add(buffer_size, tile_size)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        # Simulator does 1 cycle per output element (tile)
        return self.OTPC()

    def N_fire(self) -> sympy.Expr:
        in_stream = get_stream(self._input)
        return in_stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        in_stream = get_stream(self._input)
        assert isinstance(in_stream.stream_dtype, Buffer)
        e = sympy.Integer(1)
        for dim in in_stream.stream_dtype.shape:
            e = e * (dim.expr if isinstance(dim, DynDim) else dim)
        for rf in self.repeat_factor:
            e = e * rf
        return e



class DynStreamify(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    ref: Union[StepOps, Tuple[StepOps, int]]
    repeat_rank: int
    bufferized_rank: int
    _stream: Stream
    off_chip: bool

    # [Genghan] There is an ExpandRef hidden in the operation
    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        ref: Union[StepOps, Tuple[StepOps, int]],
        repeat_rank: int,  # Starting from this rank to rank 0, the input_stream should have 1s
        bufferized_rank: int,
        off_chip: bool = False,
    ):
        super().__init__()

        self._input = input
        self.ref = ref
        self.repeat_rank = repeat_rank
        self.bufferized_rank = bufferized_rank
        self.off_chip = off_chip

        in_stream: Stream = get_stream(input)
        ref_stream: Stream = get_stream(ref)

        assert bufferized_rank > 0, "Bufferized rank must be greater than 0."
        calc_rank = self.repeat_rank + 1
        assert (
            in_stream.shape[:-calc_rank] == ref_stream.shape[:-calc_rank]
        ), f"Shapes up to the repeat rank don't match: {in_stream.shape[: -calc_rank]} != {ref_stream.shape[: -calc_rank]}"

        assert (
            in_stream.shape[-calc_rank:] == (1,) * calc_rank
        ), f"Input stream shape must have 1s in the repeat rank dimensions {in_stream.shape[-calc_rank :]} != {(1,) * calc_rank}."

        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Buffer type."

        buffer_shape = in_stream.stream_dtype.shape
        buffer_dtype = in_stream.stream_dtype.buff_dtype

        self._stream = Stream(
            stream_dtype=buffer_dtype, shape=ref_stream.shape + buffer_shape
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(ref_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.ref == org_input:
            if get_stream(self.ref) != get_stream(new_input):
                raise ValueError("The shape of the ref stream shouldn't change")
            self.ref = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        if not self.off_chip:
            return sympy.Integer(0)

        # Get the ref stream
        ref_stream = get_stream(self.ref)

        # Calculate the product of elements in the ref stream's shape
        ref_elements = ref_stream.total_elements()

        # Get the size of the buffer (the data type of self._stream)
        buffer_size = self._stream.stream_dtype.size_in_bytes()

        return sympy.Mul(ref_elements, buffer_size)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        # Get the size of the stream's data type
        assert isinstance(
            self._stream.stream_dtype, Tile
        ), "Stream must be a Tile type."
        tile_size = self._stream.stream_dtype.size_in_bytes()

        if self.off_chip:
            return tile_size

        in_stream = get_stream(self._input)
        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Stream must be a Buffer type."
        # Calculate the size of the buffer (self._stream's datatype)
        buffer_size = in_stream.stream_dtype.size_in_bytes()

        return sympy.Add(buffer_size, tile_size)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        tile_size_bits = self._stream.stream_dtype.size_in_bytes() * 8
        scratchpad_bw = hw_config.get("scratchpad_bw", 1024)
        return sympy.ceiling(tile_size_bits / scratchpad_bw)

    def N_fire(self) -> sympy.Expr:
        ref_stream = get_stream(self.ref)
        return ref_stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        ref_node = self.ref if isinstance(self.ref, StepOps) else self.ref[0]
        return {input_node.instance_id: sympy.Integer(1), ref_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        in_stream = get_stream(self._input)
        assert isinstance(in_stream.stream_dtype, Buffer)
        e = sympy.Integer(1)
        for dim in in_stream.stream_dtype.shape:
            e = e * (dim.expr if isinstance(dim, DynDim) else dim)
        return e



class Parallelize(StepOps):
    """
    [Da,...,D0] x 1 -> [Da//par, ..., D0] x par
    b = parallelize rank
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    parallelize_rank: int
    num_consumers: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        parallelize_rank: int,
        num_consumers: int,  # this is the par factor
        switch_cycles: List[int] = None,
        write_back_mu: bool = False,
    ):
        super().__init__()
        self._input = input
        self.parallelize_rank = parallelize_rank
        self.num_consumers = num_consumers
        self.switch_cycles = (
            switch_cycles if switch_cycles is not None else [1] * num_consumers
        )
        self.write_back_mu = write_back_mu

        in_stream: Stream = get_stream(input)
        assert (
            parallelize_rank == in_stream.rank
        ), "Parallelize rank must be the same as the rank of the input stream"

        if isinstance(in_stream.shape[0], DynDim):
            # As the cases for use, we sum all the parallel regions, we simply do a truediv
            self._stream = [
                Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=(in_stream.shape[0] / num_consumers,) + in_stream.shape[1:],
                )
                for i in range(num_consumers)
            ]

        else:
            if in_stream.shape[0] % (num_consumers) != 0:
                raise NotImplementedError(
                    "The symbolic shape for (in_stream.shape[0] % (num_consumers) != 0) is not supported yet "
                )
            else:
                self._stream = [
                    Stream(
                        stream_dtype=in_stream.stream_dtype,
                        shape=(in_stream.shape[0] // num_consumers,)
                        + in_stream.shape[1:],
                    )
                    for i in range(num_consumers)
                ]

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        return self._stream[idx]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.parallelize_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.input == org_input:
            if get_stream(self.input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the input stream
        in_stream = get_stream(self._input)

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, (Tile, Select)
        ), "Input stream must be a Tile or Select type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()

        # Return the size times (num_consumers + 1) for FIFO requirements
        return sympy.Mul(in_tile_size, sympy.Integer(self.num_consumers + 1))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        total = sympy.Integer(0)
        for s in self._stream:
            total = total + s.total_elements()
        return total

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class StaticReassemble(StepOps):
    """
    Inverse of Parallelize.
    [Da//par, ..., D0] x par -> [Da,...,D0] x 1
    Merges multiple parallel streams back into a single stream.
    """

    _inputs: List[Union[StepOps, Tuple[StepOps, int]]]
    merge_rank: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        inputs: List[Union[StepOps, Tuple[StepOps, int]]],
        merge_rank: int,
        switch_cycles: List[int] = None,
        write_back_mu: bool = False,
    ):
        super().__init__()

        self._inputs = inputs
        self.merge_rank = merge_rank
        self.switch_cycles = (
            switch_cycles if switch_cycles is not None else [1] * len(inputs)
        )
        self.write_back_mu = write_back_mu

        in_streams: List[Stream] = [get_stream(input) for input in inputs]

        self._stream = Stream(
            stream_dtype=in_streams[0].stream_dtype,
            shape=(in_streams[0].shape[0] * len(in_streams),) + in_streams[0].shape[1:],
        )

        for input_node in inputs:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.add_edge(node, self)

    @property
    def stream(self) -> Stream:
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple input streams"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return self._inputs

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.merge_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        for i, input_node in enumerate(self._inputs):
            if input_node == org_input:
                if get_stream(input_node) != get_stream(new_input):
                    raise ValueError("The shape of the input stream shouldn't change")
                self._inputs[i] = new_input
                return
        raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the first input stream (all inputs should have the same datatype)
        in_stream = get_stream(self._inputs[0])

        # Check that the input stream's datatype is a Tile or Select
        assert isinstance(
            in_stream.stream_dtype, (Tile, Select)
        ), "Input stream must be a Tile or Select type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()

        # Return the size times (len(self._inputs) + 1) for FIFO requirements
        return sympy.Mul(in_tile_size, sympy.Integer(len(self._inputs) + 1))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        result = {}
        for inp in self._inputs:
            node = inp if isinstance(inp, StepOps) else inp[0]
            result[node.instance_id] = sympy.Integer(1)
        return result

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class FlatPartition(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    control: Union[StepOps, Tuple[StepOps, int]]
    num_consumers: int
    partition_rank: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        control: Union[StepOps, Tuple[StepOps, int]],
        partition_rank: int,
        switch_cycles: List[int],
        write_back_mu: bool,
        num_consumers: int,
    ):
        super().__init__()

        self._input = input
        self.control = control
        self.num_consumers = num_consumers
        self.partition_rank = partition_rank
        self.switch_cycles = switch_cycles
        self.write_back_mu = write_back_mu

        input_node = input if isinstance(input, StepOps) else input[0]
        control_node = control if isinstance(control, StepOps) else control[0]
        graph.add_edge(input_node, self)
        graph.add_edge(control_node, self)

        in_stream: Stream = get_stream(input)
        # [Genghan] A trick: StepOps should use the same control_node to align the outermost dimension
        new_names = [
            sympy.Symbol(f"{str(control_node)}_{i:03d}", integer=True, nonnegative=True)
            for i in range(num_consumers)
        ]

        self._stream = [
            Stream(
                stream_dtype=in_stream.stream_dtype,
                shape=(DynDim(new_names[i]),)
                + in_stream.shape[len(in_stream.shape) - partition_rank :],
            )
            for i in range(num_consumers)
        ]

        if isinstance(get_stream(control).stream_dtype, Select):
            assert get_stream(control).stream_dtype.total_n == num_consumers

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.control]

    def stream_idx(self, idx: int) -> Stream:
        return self._stream[idx]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.partition_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.input == org_input:
            if get_stream(self.input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.control == org_input:
            if get_stream(self.control) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.control = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the input stream
        in_stream = get_stream(self._input)

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()

        # Return the size times (num_consumers + 1) for FIFO requirements
        return sympy.Mul(in_tile_size, sympy.Integer(self.num_consumers + 1))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        total = sympy.Integer(0)
        for s in self._stream:
            total = total + s.total_elements()
        return total

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        ctrl_node = self.control if isinstance(self.control, StepOps) else self.control[0]
        return {input_node.instance_id: sympy.Integer(1), ctrl_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class EagerMerge(StepOps):
    _inputs: List[Union[StepOps, Tuple[StepOps, int]]]
    input_rank: int  # this is the merge (reassemble) rank too
    num_consumers: int
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        inputs: List[Union[StepOps, Tuple[StepOps, int]]],
        input_rank: int,  # Remove dimensions at rank larger or equal to this value
    ):
        super().__init__()

        self._inputs = inputs
        self.input_rank = input_rank
        self.num_consumers = 2

        in_streams: List[Stream] = [get_stream(input) for input in inputs]
        # assert all(
        #     stream.shape[1:] == in_streams[0].shape[1:] for stream in in_streams
        # ), "All input streams must have the same shape except the outermost dimensions."

        # True if any input streams has a dynamic outermost dimension
        has_dyndim_input = any(
            isinstance(stream.shape[0], DynDim) for stream in in_streams
        )
        merged_dim = DynDim(sympy.Integer(0)) if has_dyndim_input else 0
        merged_dim = sum([stream.shape[0] for stream in in_streams], start=merged_dim)

        stream_dtype = in_streams[0].stream_dtype
        if not all(stream.stream_dtype == stream_dtype for stream in in_streams):
            if all(isinstance(stream.stream_dtype, DynTile) for stream in in_streams):
                # Helper to compute max that works with both int and DynDim
                def symbolic_max(values):
                    # Check if any value is DynDim
                    has_dyndim = any(isinstance(v, DynDim) for v in values)
                    if has_dyndim:
                        # Convert all to sympy expressions
                        exprs = [
                            v.expr if isinstance(v, DynDim) else sympy.Integer(v)
                            for v in values
                        ]
                        return DynDim(sympy.Max(*exprs))
                    else:
                        # All are integers, use regular max
                        return max(values)

                tile_row = symbolic_max(
                    [row_i.stream_dtype.shape[0] for row_i in in_streams]
                )
                tile_col = symbolic_max(
                    [col_i.stream_dtype.shape[1] for col_i in in_streams]
                )

                stream_dtype = DynTile(
                    tile_dtype=stream_dtype.tile_dtype,
                    shape=(tile_row, tile_col),
                )

            else:
                raise ValueError(
                    "All input streams must have the same data type or be DynTile types."
                )
        data_stream = Stream(
            stream_dtype=stream_dtype,
            # stream_dtype=in_streams[0].stream_dtype,
            shape=(merged_dim,) + in_streams[0].shape[1:],
        )
        sel_stream = Stream(
            stream_dtype=MultiHot(total_n=len(in_streams)),
            shape=(merged_dim,),
        )
        self._stream = [data_stream, sel_stream]

        for input_node in inputs:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.add_edge(node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple input streams"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return self._inputs

    def stream_idx(self, idx: int) -> Stream:
        """
        idx:
        0: data stream
        1: select stream
        """
        assert idx in [0, 1], "Invalid stream index"
        return self._stream[idx]

    def data_stream(self) -> Stream:
        return self._stream[0]

    def data_tuple(self) -> Tuple[StepOps, int]:
        return (self, 0)

    def select_tuple(self) -> Tuple[StepOps, int]:
        return (self, 1)

    def select_stream(self) -> Stream:
        return self._stream[1]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.input_rank} D)"

    def replace_full_input(
        self,
        graph: MultiDiGraph,
        new_input_list: List[Union["StepOps", Tuple["StepOps", int]]],
    ):
        assert len(self._inputs) == len(
            new_input_list
        ), "The number of inputs should be the same"

        for org_node, new_node in zip(self._inputs, new_input_list):
            if get_stream(org_node).shape != get_stream(new_node).shape:
                raise ValueError("The shape of the input stream shouldn't change")

        for input_node in self._inputs:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.remove_node(node)

        self._inputs = new_input_list

        for input_node in new_input_list:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.add_edge(node, self)

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        for i, input_node in enumerate(self._inputs):
            if input_node == org_input:
                if get_stream(input_node) != get_stream(new_input):
                    raise ValueError("The shape of the input stream shouldn't change")
                self._inputs[i] = new_input
                return

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the first input stream (all inputs should have the same datatype)
        in_stream = get_stream(self._inputs[0])

        sel_stream_out: Stream = self.stream_idx(1)

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()
        sel_stream_out_size = sel_stream_out.stream_dtype.size_in_bytes()

        # Return the size times (len(self._inputs) + 1) for FIFO requirements

        return sympy.Mul(
            sympy.Add(in_tile_size, sel_stream_out_size),
            sympy.Integer(len(self._inputs) + 1),
        )

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream[0].total_elements()

    def NIT(self) -> dict:
        result = {}
        for inp in self._inputs:
            node = inp if isinstance(inp, StepOps) else inp[0]
            result[node.instance_id] = sympy.Integer(1)
        return result

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class FlatReassemble(StepOps):
    _inputs: List[Union[StepOps, Tuple[StepOps, int]]]
    control: Union[StepOps, Tuple[StepOps, int]]
    reassemble_rank: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        inputs: List[Union[StepOps, Tuple[StepOps, int]]],
        control: Union[StepOps, Tuple[StepOps, int]],
        reassemble_rank: int,  # Remove dimensions at rank larger or equal to this value
        switch_cycles: List[int],
        write_back_mu: bool,
    ):
        super().__init__()

        self._inputs = inputs
        self.control = control
        self.reassemble_rank = reassemble_rank
        self.switch_cycles = switch_cycles
        self.write_back_mu = write_back_mu

        in_streams = [get_stream(input) for input in inputs]
        assert all(
            stream.shape[len(stream.shape) - reassemble_rank :]
            == in_streams[0].shape[len(in_streams[0].shape) - reassemble_rank :]
            for stream in in_streams
        ), "All input streams must have the same shape for the last 'reassemble_rank' dimensions."
        control_stream: Stream = get_stream(control)
        new_name = DynDim(f"{str(self)}_dyn")
        self._stream = Stream(
            stream_dtype=in_streams[0].stream_dtype,
            shape=control_stream.shape
            + (new_name,)
            + in_streams[0].shape[len(in_streams[0].shape) - reassemble_rank :],
        )

        for input_node in inputs:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.add_edge(node, self)

        control_node = control if isinstance(control, StepOps) else control[0]
        graph.add_edge(control_node, self)

    @property
    def stream(self) -> Stream:
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple input streams"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return self._inputs + [self.control]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.reassemble_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        for i, input_node in enumerate(self._inputs):
            if input_node == org_input:
                if get_stream(input_node) != get_stream(new_input):
                    raise ValueError("The shape of the input stream shouldn't change")
                self._inputs[i] = new_input
                return

        if self.control == org_input:
            if get_stream(self.control) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.control = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            if self.write_back_mu:
                reassembled_elements = sympy.Integer(1)
                for dim in self._stream.shape[get_stream(self.control).rank - 1 :]:
                    if isinstance(dim, DynDim):
                        reassembled_elements *= dim.expr
                    else:
                        reassembled_elements *= dim
                return reassembled_elements * self._stream.stream_dtype.size_in_bytes()
            return sympy.Integer(0)

        # Get the first input stream (all inputs should have the same datatype)
        in_stream = get_stream(self._inputs[0])

        # Check that the input stream's datatype is a Tile
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        # Calculate the size of the input stream's datatype
        in_tile_size = in_stream.stream_dtype.size_in_bytes()

        if self.write_back_mu:
            reassembled_elements = sympy.Integer(1)
            for dim in self._stream.shape[get_stream(self.control).rank - 1 :]:
                if isinstance(dim, DynDim):
                    reassembled_elements *= dim.expr
                else:
                    reassembled_elements *= dim
            return sympy.Add(
                sympy.Mul(in_tile_size, sympy.Integer(len(self._inputs) + 1)),
                reassembled_elements * self._stream.stream_dtype.size_in_bytes(),
            )

        # Return the size times (len(self._inputs) + 1) for FIFO requirements

        return sympy.Mul(in_tile_size, sympy.Integer(len(self._inputs) + 1))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        result = {}
        for inp in self._inputs:
            node = inp if isinstance(inp, StepOps) else inp[0]
            result[node.instance_id] = sympy.Integer(1)
        ctrl_node = self.control if isinstance(self.control, StepOps) else self.control[0]
        result[ctrl_node.instance_id] = sympy.Integer(1)
        return result

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class UnaryMap(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        fn: MapFn,
        write_back_mu: bool,
        compute_bw: int,
    ):
        super().__init__()

        self._input = input
        self.fn = fn
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in_stream: Stream = get_stream(input)

        self._stream = Stream(
            stream_dtype=self.fn.apply((in_stream.stream_dtype,)),
            shape=in_stream.shape,
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} (fn: {self.fn})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self._input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get stream for input
        in_stream = get_stream(self._input)

        # Calculate tile sizes in bytes for each stream
        assert not isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        assert not isinstance(
            self._stream.stream_dtype, Buffer
        ), "Input stream must be a Tile type."

        in_tile_size = in_stream.stream_dtype.size_in_bytes()
        output_tile_size = self._stream.stream_dtype.size_in_bytes()

        return sympy.Add(in_tile_size, output_tile_size)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        in_stream = get_stream(self._input)
        flops = _compute_flops_per_tile(self.fn, [in_stream], self._stream)
        out_tile_bytes = self._stream.stream_dtype.size_in_bytes()
        return _compute_roofline(flops, self.compute_bw, out_tile_bytes, self.write_back_mu)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Accum(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    fn: AccumFn
    init_fn: InitFn
    accum_rank: int
    write_back_mu: bool
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        output_stream_dtype: Union[DynTile, Tile, Buffer, Select],
        fn: AccumFn,
        init_fn: InitFn,
        accum_rank: int,
        write_back_mu: bool,
        compute_bw: int,
    ):
        super().__init__()

        self._input = input
        self.fn = fn
        self.init_fn = init_fn
        self.accum_rank = accum_rank
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in_stream: Stream = get_stream(input)
        assert accum_rank > 0, "Accum rank must be greater than 0."

        self._stream = Stream(
            stream_dtype=output_stream_dtype,
            shape=in_stream.shape[: -self.accum_rank],
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.accum_rank} D) (fn: {self.fn})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self._input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        accumulator_size = self._stream.stream_dtype.size_in_bytes()

        in_stream_dtype = get_stream(self._input).stream_dtype
        assert not isinstance(
            in_stream_dtype, Buffer
        ), "Input stream must be a Tile type."
        input_size = in_stream_dtype.size_in_bytes()
        if not count_fifos:
            return accumulator_size

        return sympy.Add(accumulator_size, input_size)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        in_stream = get_stream(self._input)
        in_tile = in_stream.stream_dtype
        # FLOPs per accumulation step: 1 op per input element
        if isinstance(in_tile, (Tile, DynTile)):
            r, c = in_tile.shape[0], in_tile.shape[1]
            flops = r * c if isinstance(r, int) and isinstance(c, int) else r * c
        else:
            flops = sympy.Integer(1)
        out_tile_bytes = self._stream.stream_dtype.size_in_bytes()
        return _compute_roofline(sympy.Integer(flops), self.compute_bw, out_tile_bytes, self.write_back_mu)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        in_stream = get_stream(self._input)
        k = sympy.Integer(1)
        for dim in in_stream.shape[-self.accum_rank:]:
            k = k * (dim.expr if isinstance(dim, DynDim) else dim)
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: k}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Flatten(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    min_rank: int
    max_rank: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        min_rank: int,
        max_rank: int,
    ):
        super().__init__()
        self._input = input
        self.min_rank = min_rank
        self.max_rank = max_rank

        input_stream: Stream = get_stream(input)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(
                self._compute_flattened_shape(input_stream.shape, min_rank, max_rank)
            ),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    def _compute_flattened_shape(
        self, shape: Tuple[int | DynDim, ...], min_rank, max_rank
    ):
        # Convert ranks to indices (rank 0 = rightmost = highest index)
        min_index = len(shape) - 1 - max_rank  # Note: max_rank gives min_index
        max_index = len(shape) - 1 - min_rank  # Note: min_rank gives max_index

        # Validate indices
        if min_index < 0 or max_index >= len(shape) or min_index > max_index:
            raise ValueError("Invalid rank range")

        # Calculate merged dimension
        merged_dim: Union[int, DynDim] = 1
        for i in range(min_index, max_index + 1):
            if isinstance(shape[i], DynDim):
                merged_dim = shape[i] * merged_dim  # type: ignore
            else:
                merged_dim = merged_dim * shape[i]  # type: ignore

        # Build new shape
        new_shape = shape[:min_index] + (merged_dim,) + shape[max_index + 1 :]

        return new_shape

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.min_rank} D, {self.max_rank} D)"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the size of the stream's data type
        stream_size = self._stream.stream_dtype.size_in_bytes()
        return sympy.Mul(stream_size, sympy.Integer(2))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class ReshapePadStream(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    chunk_size: int
    reshape_rank: int
    num_consumers: int
    pad_fn: Optional[InitFn]
    write_back_mu: bool
    input_stream_rank: int
    _stream: List[Stream]  # data stream, pad stream
    have_pad_stream: bool

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        chunk_size: int,
        reshape_rank: int,
        write_back_mu: bool,
        pad_fn: Optional[InitFn] = None,
        have_pad_stream: bool = True,
    ):
        super().__init__()
        self._input = input
        self.chunk_size = chunk_size
        self.reshape_rank = reshape_rank
        self.num_consumers = 2
        self.pad_fn = pad_fn
        self.write_back_mu = write_back_mu
        self.have_pad_stream = have_pad_stream

        in_stream: Stream = get_stream(input)
        self.input_stream_rank = in_stream.rank

        rank_pos = in_stream.rank - reshape_rank
        # assert (
        #     isinstance(in_stream.shape[rank_pos], int)
        #     and in_stream.shape[rank_pos] % chunk_size == 0
        # ) or (
        #     reshape_rank == 0 and (pad_fn is not None or chunk_size == 1)
        # ), "The chunk size must be a divisor of the shape at the reshape rank if the rank being split is not the innermost"

        assert (
            reshape_rank >= 0 and reshape_rank <= in_stream.rank
        ), f"Reshape rank must be between 0 and {in_stream.rank}."

        if isinstance(in_stream.shape[rank_pos], DynDim):
            self._stream = [
                Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                ),
                Stream(
                    stream_dtype=Tile(tile_dtype=Bool(), shape=(1, 1)),
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                ),
            ]
        elif isinstance(in_stream.shape[rank_pos], int):
            self._stream = [
                Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                ),
                Stream(
                    stream_dtype=Tile(tile_dtype=Bool(), shape=(1, 1)),
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                ),
            ]
        else:
            raise ValueError(
                f"Unsupported shape type at rank {rank_pos}: {in_stream.shape[rank_pos]} (type: {type(in_stream.shape[rank_pos])})"
            )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        if self.have_pad_stream:
            raise NotImplementedError(
                "This property shouldn't be used for nodes with multiple output streams"
            )
        else:
            return self._stream[0]

    @property
    def stream_list(self) -> List[Stream]:
        if self.have_pad_stream:
            return self._stream
        else:
            return [self._stream[0]]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        """
        idx:
        0: data stream
        1: pad stream
        """
        assert idx in [0, 1], "Invalid stream index"
        assert idx == 0 or self.have_pad_stream, "Pad stream is not available"
        return self._stream[idx]

    def data_stream(self) -> Stream:
        return self._stream[0]

    def data_tuple(self) -> Tuple[StepOps, int]:
        return (self, 0)

    def pad_tuple(self) -> Tuple[StepOps, int]:
        assert self.have_pad_stream, "Pad stream is not available"
        return (self, 1)

    def pad_stream(self) -> Stream:
        assert self.have_pad_stream, "Pad stream is not available"
        return self._stream[1]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.reshape_rank} D) (chunk: {self.chunk_size})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the size of the stream's data type
        stream_size = self._stream[0].stream_dtype.size_in_bytes()
        return sympy.Mul(stream_size, sympy.Integer(2))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream[0].total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class Reshape(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    chunk_size: int
    reshape_rank: int
    pad_fn: Optional[InitFn]
    write_back_mu: bool
    input_stream_rank: int
    add_outer_dim: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        chunk_size: int,
        reshape_rank: int,
        write_back_mu: bool,
        add_outer_dim: bool = False,
        pad_fn: Optional[InitFn] = None,
    ):
        super().__init__()
        self._input = input
        self.chunk_size = chunk_size
        self.reshape_rank = reshape_rank
        self.pad_fn = pad_fn
        self.write_back_mu = write_back_mu

        in_stream: Stream = get_stream(input)
        self.input_stream_rank = in_stream.rank

        rank_pos = in_stream.rank - reshape_rank
        assert (
            isinstance(in_stream.shape[rank_pos], int)
            and in_stream.shape[rank_pos] % chunk_size == 0
        ) or (
            reshape_rank == 0 and (pad_fn is not None or chunk_size == 1)
        ), "The chunk size must be a divisor of the shape at the reshape rank if the rank being split is not the innermost"

        if add_outer_dim:
            assert (
                in_stream.rank == 0
            ), "Input stream rank must be 0 if add_outer_dim is True"
        self.add_outer_dim = add_outer_dim

        assert (
            reshape_rank >= 0 and reshape_rank <= in_stream.rank
        ), f"Reshape rank must be between 0 and {in_stream.rank}."

        if isinstance(in_stream.shape[rank_pos], DynDim):
            if add_outer_dim:  # this means in_stream.rank == 0
                outer_most_expr: sympy.Expr = in_stream.shape[rank_pos].expr
                # print(f"outer_most_expr: {outer_most_expr}")
                dyn_1 = DynDim(
                    Piecewise(
                        (Integer(1), outer_most_expr >= 1),
                        (Integer(0), outer_most_expr < 1),
                    )
                )
                # print(f"dyn_1: {dyn_1}")

                self._stream = Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=(dyn_1,)
                    + in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                )

            else:
                self._stream = Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                )
        elif isinstance(in_stream.shape[rank_pos], int):
            if add_outer_dim:  # this means in_stream.rank == 0
                assert (
                    in_stream.shape[rank_pos] > 0
                ), "The outer most dimension must be greater than 0 if add_outer_dim is True for static case"
                self._stream = Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=(1,)
                    + in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                )

            else:
                self._stream = Stream(
                    stream_dtype=in_stream.stream_dtype,
                    shape=in_stream.shape[:rank_pos]
                    + ((in_stream.shape[rank_pos] + chunk_size - 1) // chunk_size,)
                    + (chunk_size,)
                    + in_stream.shape[(rank_pos + 1) :],
                )
        else:
            raise ValueError(
                f"Unsupported shape type at rank {rank_pos}: {in_stream.shape[rank_pos]} (type: {type(in_stream.shape[rank_pos])})"
            )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id} ({self.reshape_rank} D) (chunk: {self.chunk_size})"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        if not count_fifos:
            return sympy.Integer(0)

        # Get the size of the stream's data type
        stream_size = self._stream.stream_dtype.size_in_bytes()
        return sympy.Mul(stream_size, sympy.Integer(2))

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)



class RetileStreamify(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    split_row: bool
    filter_mask: bool
    chunk: int
    num_chunks: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        split_row: bool,
        filter_mask: bool = False,
        chunk: int = 1,
    ):
        super().__init__()
        self._input = input
        self.split_row = split_row
        self.filter_mask = filter_mask
        self.chunk = chunk
        in_stream: Stream = get_stream(input)

        assert isinstance(
            in_stream.stream_dtype, (Tile, DynTile)
        ), "Input stream must be a Tile type."

        in_stream_tile: Union[Tile, DynTile] = in_stream.stream_dtype

        if split_row:
            assert chunk == 1 or (
                in_stream_tile.shape[0] % chunk == 0
            ), f"Tile row size {in_stream_tile.shape[0]} must be divisible by chunk size {chunk}."
            num_chunks = in_stream_tile.shape[0] // chunk
            self.num_chunks = num_chunks

            if filter_mask:
                output_stream_shape = in_stream.shape[:-1] + (
                    DynDim(f"{str(self)}_dyn"),
                )
                output_stream_dtype = Tile(
                    shape=(chunk, in_stream_tile.shape[1]),
                    tile_dtype=in_stream_tile.tile_dtype,
                )

            else:
                output_stream_shape = in_stream.shape[:-1] + (
                    in_stream.shape[-1] * num_chunks,
                )
                output_stream_dtype = Tile(
                    shape=(chunk, in_stream_tile.shape[1]),
                    tile_dtype=in_stream_tile.tile_dtype,
                )

        else:
            assert (
                in_stream_tile.shape[1] % chunk == 0
            ), f"Tile col size {in_stream_tile.shape[1]} must be divisible by chunk size {chunk}."
            num_chunks = in_stream_tile.shape[1] // chunk
            self.num_chunks = num_chunks

            if filter_mask:
                output_stream_shape = in_stream.shape[:-1] + (
                    DynDim(f"{str(self)}_dyn"),
                )
                output_stream_dtype = Tile(
                    shape=(in_stream.stream_dtype.shape[0], chunk),
                    tile_dtype=in_stream_tile.tile_dtype,
                )
            else:
                output_stream_shape = in_stream.shape[:-1] + (
                    in_stream.shape[-1] * num_chunks,
                )
                output_stream_dtype = Tile(
                    shape=(in_stream.stream_dtype.shape[0], chunk),
                    tile_dtype=in_stream_tile.tile_dtype,
                )

        self._stream = Stream(
            stream_dtype=output_stream_dtype,
            shape=output_stream_shape,
        )
        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        return self._stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(self.num_chunks)



class FlatmapFilterRowStreamify(StepOps):
    """
    This is an instance of the Flatmap operator that takes in the filter + row-wise streamify function.
    The filter function filters the rows in the tile based on the mask stream.


    input data stream shape: [..., D1, D0] (tile shape: [R, C])
    mask stream shape:       [..., D1, D0] (tile shape: [R, 1], dtype = Bool)
    output stream shape:     [..., D1, D0'] (tile shape: [1, C])
    * D0' denotes a new dynamic dimension as this can be a ragged dimension and
      is determined by the content of the mask stream.

    function signature: (Tile (data), Tile (mask)) => Stream<Tile,s>
    * s is the stop token level attached to the input data tile
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    mask: Union[StepOps, Tuple[StepOps, int]]
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        mask: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()
        self._input = input
        self.mask = mask
        in_stream: Stream = get_stream(input)
        mask_stream: Stream = get_stream(mask)

        assert isinstance(
            in_stream.stream_dtype, (Tile, DynTile)
        ), "Input stream must be a Tile type."

        assert isinstance(
            mask_stream.stream_dtype, (Tile, DynTile)
        ), "Mask stream must be a Tile type."
        assert isinstance(
            mask_stream.stream_dtype.tile_dtype, Bool
        ), "Mask stream's tile type must be a Bool type."

        in_stream_tile: Union[Tile, DynTile] = in_stream.stream_dtype

        self._stream = Stream(
            stream_dtype=Tile(
                shape=(1, in_stream_tile.shape[1]),
                tile_dtype=in_stream_tile.tile_dtype,
            ),
            shape=in_stream.shape[:-1] + (DynDim(f"{str(self)}_dyn"),),
        )
        input_node = input if isinstance(input, StepOps) else input[0]
        mask_node = mask if isinstance(mask, StepOps) else mask[0]
        graph.add_edge(input_node, self)
        graph.add_edge(mask_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple input streams"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.mask]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.mask == org_input:
            if get_stream(self.mask) != get_stream(new_input):
                raise ValueError("The shape of the mask stream shouldn't change")
            self.mask = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        in_stream = get_stream(self._input)
        return in_stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        mask_node = self.mask if isinstance(self.mask, StepOps) else self.mask[0]
        return {input_node.instance_id: sympy.Integer(1), mask_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        in_tile = get_stream(self._input).stream_dtype
        return sympy.Integer(in_tile.shape[0])



class FlatmapCounter(StepOps):
    """
    scalar_i32 -> Stream<scalar_i32,1>

    example) 3 -> 0 1 2 S1
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()
        self._input = input
        in_stream: Stream = get_stream(input)

        self._stream = Stream(
            stream_dtype=Tile(
                shape=(1, 1), tile_dtype=in_stream.stream_dtype.tile_dtype
            ),
            shape=in_stream.shape + (DynDim(f"{str(self)}"),),
        )
        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)

    def T_fire(self, hw_config: dict) -> sympy.Expr:
        return sympy.Integer(0)

    def N_fire(self) -> sympy.Expr:
        in_stream = get_stream(self._input)
        return in_stream.total_elements()

    def NIT(self) -> dict:
        input_node = self._input if isinstance(self._input, StepOps) else self._input[0]
        return {input_node.instance_id: sympy.Integer(1)}

    def OTPC(self) -> sympy.Expr:
        return sympy.Integer(1)

