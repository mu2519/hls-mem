from veriloggen.core import vtypes
from veriloggen.core.module import Module
from veriloggen.seq.seq import Seq, make_condition
from veriloggen.fsm.fsm import FSM
from .ttypes import _MutexFunction
from .ram import RAM
from .axim import AXIM


def add_cond(tgt: vtypes._Variable, cond):
    cond = make_condition(cond)
    if tgt.assign_value is None:
        tgt.assign(cond)
    else:
        tgt.assign_value.statement.right = vtypes.Lor(cond, tgt.assign_value.statement.right)


class PIPO(RAM):
    __intrinsics__ = (
        'push', 'pop',
        'wait_not_empty', 'wait_not_full',
        'dma_read', 'dma_write',
        'read_producer', 'write_producer',
        'read_consumer', 'write_consumer',
    ) + _MutexFunction.__intrinsics__

    def __init__(
        self,
        m: Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int,
        addrwidth: int,
        numports: int = 1,
        length: int = 2
    ):
        # TODO: support non power of two length
        if length & (length - 1) != 0:
            raise ValueError('`length` must be a power of two')

        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst
        self.datawidth = datawidth
        self.addrwidth = addrwidth
        self.numports = numports
        self.length = length

        self.rams = [RAM(m, name + '_ram_' + str(i), clk, rst, datawidth, addrwidth, numports)
                     for i in range(2)]
        self.head = m.Reg(name + '_head', (length - 1).bit_length(), initval=0)
        self.tail = m.Reg(name + '_tail', (length - 1).bit_length(), initval=0)
        self.occupancy = m.Reg(name + '_occupancy', length.bit_length(), initval=0)

        self.seq = Seq(m, name + '_seq', clk, rst)
        self.seq.add_reset(self.head)
        self.seq.add_reset(self.tail)

        # synchronization
        self.inc_occ = m.Wire(name + '_inc_occ')
        self.dec_occ = m.Wire(name + '_dec_occ')
        self.seq.If(self.inc_occ, vtypes.Not(self.dec_occ))(
            self.occupancy.inc()
        ).Elif(vtypes.Not(self.inc_occ), self.dec_occ)(
            self.occupancy.dec()
        )

        self.mutex = None

    def _read_rtl(self, addr, port: int, cond, ptr) -> tuple[vtypes.Wire, vtypes.Wire]:
        data_list = []
        valid_list = []
        for i in range(self.length):
            data, valid = self.rams[i].read_rtl(addr, port, vtypes.Ands(cond, ptr == i))
            data_list.append(data)
            valid_list.append(valid)

        data_pat = [(ptr == i, data_list[i]) for i in range(self.length)]
        data_pat.append((None, vtypes.IntX()))
        data_wire = self.m.TmpWire(self.datawidth, prefix='read_rtl_data')
        data_wire.assign(vtypes.PatternMux(data_pat))

        valid_pat = [(ptr == i, valid_list[i]) for i in range(self.length)]
        valid_pat.append((None, vtypes.IntX()))
        valid_wire = self.m.TmpWire(prefix='read_rtl_valid')
        valid_wire.assign(vtypes.PatternMux(valid_pat))

        return data_wire, valid_wire

    def _write_rtl(self, addr, data, port: int, cond, ptr):
        for i in range(self.length):
            self.rams[i].write_rtl(addr, data, port, vtypes.Ands(cond, ptr == i))

    def read_producer_rtl(self, addr, port=0, cond=None) -> tuple[vtypes.Wire, vtypes.Wire]:
        return self._read_rtl(addr, port, cond, self.tail)

    def write_producer_rtl(self, addr, data, port=0, cond=None):
        self._write_rtl(addr, data, port, cond, self.tail)

    def read_consumer_rtl(self, addr, port=0, cond=None) -> tuple[vtypes.Wire, vtypes.Wire]:
        return self._read_rtl(addr, port, cond, self.head)

    def write_consumer_rtl(self, addr, data, port=0, cond=None):
        self._write_rtl(addr, data, port, cond, self.head)

    def read_burst_rtl(self, addr, port=0, cond=None) -> tuple[vtypes.Wire, vtypes.Wire]:
        return self.read_consumer_rtl(addr, port, cond)

    def write_burst_rtl(self, addr, data, port=0, cond=None):
        self.write_producer_rtl(addr, data, port, cond)

    def _read(self, fsm: FSM, addr, port, ptr) -> vtypes.Reg:
        port = vtypes.to_int(port)
        data, valid = self._read_rtl(addr, port, fsm.here, ptr)
        data_reg = self.m.TmpReg(self.datawidth, prefix='read_data')
        fsm.If(valid)(
            data_reg(data)
        )
        fsm.If(valid).goto_next()
        return data_reg

    def _write(self, fsm: FSM, addr, data, port, ptr):
        port = vtypes.to_int(port)
        self._write_rtl(addr, data, port, fsm.here, ptr)
        fsm.goto_next()

    def read_producer(self, fsm: FSM, addr, port=0) -> vtypes.Reg:
        return self._read(fsm, addr, port, self.tail)

    def write_producer(self, fsm: FSM, addr, data, port=0):
        self._write(fsm, addr, data, port, self.tail)

    def read_consumer(self, fsm: FSM, addr, port=0) -> vtypes.Reg:
        return self._read(fsm, addr, port, self.head)

    def write_consumer(self, fsm: FSM, addr, data, port=0):
        self._write(fsm, addr, data, port, self.head)

    def push(self, fsm: FSM):
        self.seq.If(fsm.here)(
            self.tail.inc()
        )
        add_cond(self.inc_occ, fsm.here)
        fsm.goto_next()

    def pop(self, fsm: FSM):
        self.seq.If(fsm.here)(
            self.head.inc()
        )
        add_cond(self.dec_occ, fsm.here)
        fsm.goto_next()

    @property
    def empty(self):
        return self.occupancy == 0

    @property
    def full(self):
        return self.occupancy == self.length

    def wait_not_empty(self, fsm: FSM):
        fsm.If(vtypes.Not(self.empty)).goto_next()

    def wait_not_full(self, fsm: FSM):
        fsm.If(vtypes.Not(self.full)).goto_next()

    def dma_read(self, fsm: FSM, axi: AXIM, local_addr, global_addr,
                 local_size, local_stride=1, port=0):
        axi.lock(fsm)
        axi.dma_read(fsm, self, local_addr, global_addr, local_size, local_stride, port)
        axi.unlock(fsm)

    def dma_write(self, fsm: FSM, axi: AXIM, local_addr, global_addr,
                  local_size, local_stride=1, port=0):
        axi.lock(fsm)
        axi.dma_write(fsm, self, local_addr, global_addr, local_size, local_stride, port)
        axi.unlock(fsm)

    # invalidate some methods from the parent class `RAM`
    def connect_rtl(self, *args, **kwargs):
        raise TypeError('`connect_rtl` method is no longer provided')

    def has_enable(self, *args, **kwargs):
        raise TypeError('`has_enable` method is no longer provided')

    def __getitem__(self, key):
        raise TypeError('no longer subscriptable')

    def read(self, *args, **kwargs):
        raise TypeError('`read` method is no longer provided')

    def write(self, *args, **kwargs):
        raise TypeError('`write` method is no longer provided')

    def read_rtl(self, *args, **kwargs):
        raise TypeError('`read_rtl` method is no longer provided')

    def write_rtl(self, *args, **kwargs):
        raise TypeError('`write_rtl` method is no longer provided')

    def read_low_priority(self, *args, **kwargs):
        raise TypeError('`read_low_priority` method is no longer provided')
