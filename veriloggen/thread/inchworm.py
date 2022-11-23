from veriloggen.core import vtypes, module
from veriloggen.core.vtypes import Posedge, If, Land, Lor, Ulnot, Cond
from veriloggen.fsm.fsm import FSM, TmpFSM
from .ram import RAM
from .axim import AXIM


"""
Synchronization issues

read <-> write: synchronized by caller
enqueue, dequeue <-> read, write: no need to synchronize
enqueue <-> dequeue: synchronized by callee
rebase <-> read, write, dequeue: synchronized by caller
rebase <-> enqueue: synchronized by callee
"""


# TODO: add forwarding logic to RAM
class Inchworm:
    __intrinsics__ = ('enqueue', 'dequeue', 'read', 'write', 'rebase')

    def __init__(
        self,
        m: module.Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int = 32,
        addrwidth: int = 10
    ):
        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst
        self.datawidth = datawidth
        self.addrwidth = addrwidth

        self.ram = RAM(m, name, clk, rst, datawidth, addrwidth, numports=2)

        self.base = m.Reg(name + '_base', addrwidth, signed=False, initval=0)  # base of index
        self.limit = m.Reg(name + '_limit', addrwidth + 1, signed=False, initval=0)  # limit of index
        self.head = m.Reg(name + '_head', addrwidth, signed=False, initval=0)  # head of queue
        self.occupancy = m.Reg(name + '_occupancy', addrwidth + 1, signed=False, initval=0)  # occupancy of queue
        self.size = 1 << addrwidth  # physical size of RAM

        self.inc_occ = m.Wire(name + '_inc_occ', 1, signed=False)
        self.dec_occ = m.Wire(name + '_dec_occ', 1, signed=False)
        m.Always(Posedge(clk))(
            If(rst)(
                self.occupancy(0)
            ).Else(
                If(Land(self.inc_occ, Ulnot(self.dec_occ)))(
                    self.occupancy.inc()
                ).Elif(Land(Ulnot(self.inc_occ), self.dec_occ))(
                    self.occupancy.dec()
                )
            )
        )

        self.set_lim = m.Wire(name + '_set_lim', 1, signed=False)
        self.inc_lim = m.Wire(name + '_inc_lim', 1, signed=False)
        m.Always(Posedge(clk))(
            If(rst)(
                self.limit(0)
            ).Else(
                If(Land(self.set_lim, self.inc_lim))(
                    self.limit(self.occupancy + 1)
                ).Elif(self.set_lim)(
                    self.limit(self.occupancy)
                ).Elif(self.inc_lim)(
                    self.limit.inc()
                )
            )
        )

    @property
    def empty(self):
        return self.occupancy == 0

    @property
    def full(self):
        return self.occupancy == self.size

    @property
    def vacancy(self):
        return self.size - self.occupancy

    @property
    def tail(self):
        return self.head + self.occupancy

    def enqueue(self, fsm: FSM, data: vtypes.IntegralType) -> None:
        """ this method does not check if queue is full """
        self.ram.write_rtl(self.tail, data, port=1, cond=fsm.here)
        self._add_assign(self.inc_occ, fsm.here)
        self._add_assign(self.inc_lim, fsm.here)
        fsm.goto_next()

    # used for DMA and called by AXIM._synthesize_read_data_fsm_*
    # created based on RAM.write_burst
    # unused parameters: addr, stride, blocksize, wlast, wquit, port
    def enqueue_for_dma(self, addr, stride, length, blocksize, wdata, wvalid, wlast=False, wquit=False, port=0, cond=None) -> None:
        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='enqueue_for_dma_fsm')
        length_reg = self.m.TmpReg(vtypes.get_width(length), initval=0, prefix='enqueue_for_dma_length')

        fsm(
            length_reg(length)
        )
        fsm.If(cond, length > 0).goto_next()

        self.ram.write_rtl(self.tail, wdata, port=1, cond=Land(fsm.here, wvalid))
        self._add_assign(self.inc_occ, Land(fsm.here, wvalid))
        self._add_assign(self.inc_lim, Land(fsm.here, wvalid))
        fsm.If(wvalid)(
            length_reg.dec()
        )
        fsm.If(wvalid, length_reg <= 1).goto_init()

    def dequeue(self, fsm: FSM, cond: vtypes.IntegralType | None = None) -> None:
        """ this method does not check if queue is empty """
        if cond is None:
            self._add_assign(self.dec_occ, fsm.here)
            fsm(
                self.head.inc()
            )
        else:
            self._add_assign(self.dec_occ, Land(fsm.here, cond))
            fsm.If(cond)(
                self.head.inc()
            )
        fsm.goto_next()

    def read(self, fsm: FSM, index: vtypes.IntegralType) -> vtypes.Reg:
        addr = self.base + index
        cond = Land(fsm.here, self.limit > index)
        data, valid = self.ram.read_rtl(addr, port=0, cond=cond)
        data_reg = self.m.TmpReg(self.datawidth, signed=False, initval=0, prefix='read_data')
        fsm.If(valid)(
            data_reg(data)
        )
        fsm.If(valid).goto_next()
        return data_reg

    def write(self, fsm: FSM) -> None:
        pass

    def rebase(self, fsm: FSM) -> None:
        self._add_assign(self.set_lim, fsm.here)
        fsm(
            self.base(self.head)
        )
        fsm.goto_next()

    def _add_assign(self, var: vtypes.Wire, val: vtypes.IntegralType) -> None:
        if var.assign_value is None:
            var.assign(val)
        else:
            var.assign_value.statement.right = Lor(val, var.assign_value.statement.right)


__intrinsics__ = ('prefetch_dma_read',)


def prefetch_dma_read(fsm: FSM, axi: AXIM, ram: Inchworm, global_addr, region_size, block_size):
    if not isinstance(axi, AXIM):
        raise TypeError('AXIM object is required for AXI module')
    if not isinstance(ram, Inchworm):
        raise TypeError('Inchworm object is required for RAM module')

    if fsm.m is not axi.m:
        raise ValueError('different modules are not allowed')
    m = fsm.m

    block_size_in_words = block_size
    word_size = ram.datawidth // 8
    # n & (n - 1) = 0 iff n = 2^k
    # test if n is a power of two
    if word_size & (word_size - 1) == 0:
        # n.bit_length() - 1 gives k for n = 2^k
        block_size_in_bytes = block_size_in_words << (word_size.bit_length() - 1)
    else:
        block_size_in_bytes = block_size_in_words * word_size

    addr = m.TmpReg(axi.addrwidth, signed=False, initval=0, prefix='prefetch_dma_addr')  # address in bytes
    size = m.TmpReg(axi.addrwidth + 1, signed=False, initval=0, prefix='prefetch_dma_size')  # size in words

    next_transfer_size = m.TmpWire(axi.addrwidth + 1, signed=False, prefix='prefetch_dma_read_next_transfer_size')
    next_transfer_size.assign(Cond(size <= block_size_in_words, size, block_size_in_words))

    fsm(
        addr(global_addr),
        size(region_size)
    )
    fsm.goto_next()

    loop_cond_check_count = fsm.current
    fsm.inc()
    loop_body_begin_count = fsm.current
    fsm.If(ram.vacancy >= next_transfer_size).goto_next()
    axi.dma_read(fsm, ram, 0, addr, next_transfer_size, ram_method=ram.enqueue_for_dma)
    loop_body_end_count = fsm.current
    fsm(
        addr.add(block_size_in_bytes),
        size.sub(next_transfer_size)
    )
    fsm.inc()
    loop_exit_count = fsm.current

    fsm.goto_from(loop_body_end_count, loop_cond_check_count)
    fsm.goto_from(loop_cond_check_count, loop_body_begin_count, size > 0, loop_exit_count)
