from collections import defaultdict
from collections.abc import Mapping, Callable
from typing import Any

from veriloggen.core.module import Module
from veriloggen.core import vtypes
from .ram import RAM
from veriloggen.seq.seq import Seq, make_condition
from veriloggen.fsm.fsm import FSM, TmpFSM
from .axim import AXIM


def make_flag(m: Module, name: str) -> vtypes.Wire:
    """ used in conjunction with `add_cond` """
    wire = m.Wire(name)
    wire.assign(vtypes.Int(0))
    return wire


def make_mux_data(m: Module, name: str, width: int) -> vtypes.Wire:
    """ used in conjunction with `add_mux` """
    wire = m.Wire(name, width)
    wire.assign(vtypes.Int("'hx"))
    return wire


def equal_intx(num: Any) -> bool:
    return (isinstance(num, vtypes.Int) and
            isinstance(num.value, str) and
            num.value == 'x')


def equal_int0(num: Any) -> bool:
    if isinstance(num, int):
        return num == 0
    elif isinstance(num, vtypes.Int):
        return isinstance(num.value, int) and num.value == 0
    else:
        return False


def add_mux(tgt: vtypes.Wire, cond, val):
    cond = make_condition(cond)
    if equal_intx(tgt.assign_value.statement.right):
        tgt.assign_value.statement.right = val
    else:
        tgt.assign_value.statement.right = vtypes.Mux(cond, val, tgt.assign_value.statement.right)


def add_cond(tgt: vtypes.Wire, cond):
    cond = make_condition(cond)
    if equal_int0(tgt.assign_value.statement.right):
        tgt.assign_value.statement.right = cond
    else:
        tgt.assign_value.statement.right = vtypes.Ors(cond, tgt.assign_value.statement.right)


class BuffetBase:
    __intrinsics__ = ('read', 'write', 'release', 'rebase')

    def __init__(
        self,
        m: Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int,
        addrwidth: int,
        initval: Mapping[str, int],
    ):
        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst
        self.datawidth = datawidth
        self.addrwidth = addrwidth

        self.seq = Seq(m, name + '_seq', clk, rst)

        # the underlying RAM
        self.ram = RAM(m, name + '_ram', clk, rst, datawidth, addrwidth, numports=2)

        # fundamental registers
        initval = defaultdict(int, initval)
        self.base = m.Reg(name + '_base', addrwidth, initval=initval['base'])
        self.limit = m.Reg(name + '_limit', addrwidth + 2, initval=initval['limit'])
        self.head = m.Reg(name + '_head', addrwidth, initval=initval['head'])
        self.tail = m.Reg(name + '_tail', addrwidth, initval=initval['tail'])
        self.occupancy = m.Reg(name + '_occupancy', addrwidth + 1, initval=initval['occupancy'])
        self.credit = m.Reg(name + '_credit', addrwidth + 1, initval=initval['credit'])

        # synchronization

        # synchronize `credit`
        self.credit_inc_flag = make_flag(m, name + '_credit_inc_flag')
        self.credit_dec_flag = make_flag(m, name + '_credit_dec_flag')
        self.credit_dec_num = make_mux_data(m, name + '_credit_dec_num', addrwidth + 1)
        self.seq.If(self.credit_dec_flag, self.credit_inc_flag)(
            self.credit(self.credit - self.credit_dec_num + 1)
        ).Elif(self.credit_dec_flag)(
            self.credit.sub(self.credit_dec_num)
        ).Elif(self.credit_inc_flag)(
            self.credit.inc()
        )

        # synchronize `occupancy`
        self.dec_occ = make_flag(m, name + '_dec_occ')

        # synchronize `limit`
        self.set_lim = make_flag(m, name + '_set_lim')

    def read(self, fsm: FSM, index: vtypes.IntegralType) -> vtypes.Reg:
        data, valid = self.ram.read_rtl(self.base + index, port=0,
                                        cond=(fsm.here, index < self.limit))
        data_reg = self.m.TmpReg(self.datawidth, prefix='read_data')
        fsm.If(valid)(
            data_reg(data)
        )
        fsm.If(valid).goto_next()
        return data_reg

    def write(self, fsm: FSM, index: vtypes.IntegralType, data: vtypes.IntegralType):
        self.ram.write_rtl(self.base + index, data, port=0, cond=(fsm.here, index < self.limit))
        fsm.If(index < self.limit).goto_next()

    def release(self, fsm: FSM):
        self.release_rtl(fsm.here)
        fsm.goto_next()

    def release_rtl(self, *cond):
        add_cond(self.dec_occ, cond)
        add_cond(self.credit_inc_flag, cond)
        self.seq.If(cond)(
            self.head.inc()
        )

    def rebase(self, fsm: FSM):
        add_cond(self.set_lim, fsm.here)
        self.seq.If(fsm.here)(
            self.base(self.head)
        )
        fsm.goto_next()

    def _id(self):
        return id(self)

    def dma(self, fsm: FSM, axi: AXIM, addr, size, block_size,
            axi_method: Callable[[Any, Any], None]):
        """
        addr: global address in bytes
        size: local size in words
        block_size: local block size in words
        axi_method: a callable object such that
                    the first argument is global address and
                    the second argument is local size
        """

        addr_reg = self.m.TmpReg(
            axi.addrwidth, prefix='dma_addr_reg')  # in bytes
        transferred_size = self.m.TmpReg(
            self.addrwidth + 1, prefix='dma_transferred_size')  # in words
        remaining_size = self.m.TmpReg(
            self.addrwidth + 1, prefix='dma_remaining_size')  # in words

        transferable_size = self.m.TmpWire(
            self.addrwidth + 1, prefix='dma_transferable_size')  # in words
        transferable_size.assign(vtypes.Min(self.credit, remaining_size))

        wire_to_adjust_width = self.m.TmpWire(
            axi.addrwidth, prefix='dma_wire_to_adjust_width')
        wire_to_adjust_width.assign(transferred_size)

        transferred_size_in_bytes = self.m.TmpWire(
            axi.addrwidth, prefix='dma_transferred_size_in_bytes')  # in bytes
        if not isinstance(self.datawidth, int):
            raise TypeError
        if self.datawidth < 8 or self.datawidth % 8 != 0:
            raise ValueError
        word_size = self.datawidth // 8
        # n & (n - 1) = 0 iff n = 2^k
        if word_size & (word_size - 1) == 0:
            # n.bit_length() - 1 gives k if n = 2^k
            transferred_size_in_bytes.assign(
                wire_to_adjust_width << (word_size.bit_length() - 1)
            )
        else:
            transferred_size_in_bytes.assign(wire_to_adjust_width * word_size)

        transferal_cond = self.m.TmpWire(prefix='dma_transferal_cond')
        transferal_cond.assign(
            vtypes.Lor(self.credit >= remaining_size, self.credit >= block_size)
        )

        loop_start_count = fsm.current
        fsm(
            addr_reg(addr),
            remaining_size(size)
        )
        fsm.inc()
        loop_body_begin_count = fsm.current
        add_cond(self.credit_dec_flag, (fsm.here, transferal_cond))
        add_mux(self.credit_dec_num, fsm.here, transferable_size)
        fsm.If(transferal_cond)(
            transferred_size(transferable_size),
            remaining_size.sub(transferable_size)
        )
        fsm.If(transferal_cond).goto_next()
        axi_method(addr_reg, transferred_size)
        loop_body_end_count = fsm.current
        fsm(
            addr_reg.add(transferred_size_in_bytes)
        )
        fsm.inc()
        loop_exit_count = fsm.current
        fsm.goto_from(loop_start_count, loop_body_begin_count,
                      size > 0, loop_exit_count)
        fsm.goto_from(loop_body_end_count, loop_body_begin_count,
                      remaining_size > 0, loop_exit_count)


class BuffetRead(BuffetBase):
    __intrinsics__ = ('dma_read',) + BuffetBase.__intrinsics__

    def __init__(
        self,
        m: Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int,
        addrwidth: int,
    ):
        size = 1 << addrwidth
        super().__init__(m, name, clk, rst, datawidth, addrwidth,
                         {'credit': size})

        # synchronization

        # synchronize `occupancy`
        self.inc_occ = m.Reg(name + '_inc_occ', initval=0)
        self.seq(self.inc_occ(0))  # set default
        self.seq.If(self.inc_occ, vtypes.Not(self.dec_occ))(
            self.occupancy.inc()
        ).Elif(vtypes.Not(self.inc_occ), self.dec_occ)(
            self.occupancy.dec()
        )

        # synchronize `limit`
        self.inc_lim = m.Reg(name + '_inc_lim', initval=0)
        self.seq(self.inc_lim(0))  # set default
        self.seq.If(self.set_lim, self.inc_lim)(
            self.limit(self.occupancy + 1)
        ).Elif(self.set_lim)(
            self.limit(self.occupancy)
        ).Elif(self.inc_lim)(
            self.limit.inc()
        )

    def callback_for_dma_read(
        self, addr, stride, length, blocksize,
        wdata, wvalid, wlast=False, wquit=False, port=0, cond=None
    ):
        """
        Correspondance to AXI DMA:
            addr <-> local_addr (local address)
            stride <-> local_stride (local stride)
            length <-> local_size (local size)
            blocksize <-> local_blocksize (local blocksize)
            wdata <-> rdata (data signal in read data channel)
            wvalid <-> rvalid (valid signal in read data channel)
        Unused parameters:
            addr, stride, blocksize, wlast, wquit, port
        """

        fsm = TmpFSM(self.m, self.clk, self.rst,
                     prefix='callback_for_dma_read_fsm')
        length_reg = self.m.TmpReg(self.addrwidth + 1,
                                   prefix='callback_for_dma_read_length_reg')

        fsm(
            length_reg(length)
        )
        fsm.If(cond, length > 0).goto_next()

        self.ram.write_rtl(self.tail, wdata, port=1, cond=(fsm.here, wvalid))
        self.seq.If(fsm.here, wvalid)(
            self.tail.inc(),
            self.inc_occ(1),
            self.inc_lim(1)
        )
        fsm.If(wvalid)(
            length_reg.dec()
        )
        fsm.If(wvalid, length_reg <= 1).goto_init()

    def dma_read(self, fsm: FSM, axi: AXIM, addr, size, block_size):
        def axi_method(global_addr, local_size):
            axi.dma_read_async(fsm, self, 0, global_addr, local_size,
                               ram_method=self.callback_for_dma_read)
        self.dma(fsm, axi, addr, size, block_size, axi_method)


class BuffetWrite(BuffetBase):
    __intrinsics__ = ('dma_write',) + BuffetBase.__intrinsics__

    def __init__(
        self,
        m: Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int,
        addrwidth: int,
    ):
        size = 1 << addrwidth
        super().__init__(m, name, clk, rst, datawidth, addrwidth,
                         {'limit': size, 'occupancy': size})

        # synchronization

        # synchronize `occupancy`
        self.inc_occ = make_flag(m, name + '_inc_occ')
        self.seq.If(self.inc_occ, vtypes.Not(self.dec_occ))(
            self.occupancy.inc()
        ).Elif(vtypes.Not(self.inc_occ), self.dec_occ)(
            self.occupancy.dec()
        )

        # synchronize `limit`
        self.inc_lim = make_flag(m, name + '_inc_lim')
        self.seq.If(self.set_lim, self.inc_lim)(
            self.limit(self.occupancy + 1)
        ).Elif(self.set_lim)(
            self.limit(self.occupancy)
        ).Elif(self.inc_lim)(
            self.limit.inc()
        )

    # unused parameters: addr, stride, blocksize, rquit, port
    def callback_for_dma_write(
        self, addr, stride, length, blocksize,
        rready, rquit=False, port=0, cond=None
    ) -> tuple[vtypes.Wire, vtypes.Reg, vtypes.Reg]:
        """
        Correspondance to AXI DMA:
            addr <-> local_addr
            stride <-> local_stride
            length <-> local_size
            blocksize <-> local_blocksize
            rdata <-> wdata (data signal in write data channel)
            rlast <-> wlast (last signal in write data channel)
        Unused parameters:
            addr, stride, blocksize, rquit, port
        """

        fsm = TmpFSM(self.m, self.clk, self.rst,
                     prefix='callback_for_dma_write_fsm')
        length_reg = self.m.TmpReg(self.addrwidth + 1,
                                   prefix='callback_for_dma_write_length_reg')
        rvalid = self.m.TmpReg(prefix='callback_for_dma_write_rvalid')
        rlast = self.m.TmpReg(prefix='callback_for_dma_write_rlast')

        fsm(
            length_reg(length),
            rvalid(0),
            rlast(0)
        )
        fsm.If(cond, length > 0).goto_next()

        renable = vtypes.Ands(fsm.here, vtypes.Ors(vtypes.Not(rvalid), rready))
        rdata, _ = self.ram.read_rtl(self.tail, port=1, cond=renable)
        rdata_wire = self.m.TmpWireLike(
            rdata, prefix='callback_for_dma_write_rdata_wire')
        rdata_wire.assign(rdata)

        self.seq.If(fsm.here, rready, length_reg > 0)(
            self.tail.inc()
        )
        add_cond(self.inc_occ, (fsm.here, rready, length_reg > 0))
        add_cond(self.inc_lim, (fsm.here, rready, length_reg > 0))

        fsm.If(rready, length_reg > 0)(
            length_reg.dec(),
            rvalid(1)
        )
        fsm.If(rready, length_reg <= 1)(
            rlast(1)
        )
        fsm.If(rlast, rvalid, rready)(
            rvalid(0),
            rlast(0)
        )
        fsm.If(rlast, rvalid, rready).goto_init()

        return rdata_wire, rvalid, rlast

    def dma_write(self, fsm: FSM, axi: AXIM, addr, size, block_size):
        def axi_method(global_addr, local_size):
            axi.dma_write_async(fsm, self, 0, global_addr, local_size,
                                ram_method=self.callback_for_dma_write)
        self.dma(fsm, axi, addr, size, block_size, axi_method)
