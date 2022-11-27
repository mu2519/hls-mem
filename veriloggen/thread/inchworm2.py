from typing import Literal

from veriloggen.core import module
from veriloggen.core import vtypes
from veriloggen.seq.seq import Seq, make_condition
from veriloggen.fsm.fsm import FSM, TmpFSM
from .ram import RAM


# TODO: mutex for enqueue and dequeue (avoid memory port collision)
class Inchworm:
    __intrinsics__ = ('dequeue', 'release', 'enqueue', 'read', 'write', 'rebase')

    def __init__(
        self,
        m: module.Module,
        name: str,
        clk: vtypes._Variable,
        rst: vtypes._Variable,
        datawidth: int,
        addrwidth: int,
        mode: Literal['ro', 'wo', 'rw'] = 'rw'
    ):
        if mode not in ['ro', 'wo', 'rw']:
            raise ValueError('Invalid operation mode')

        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst
        self.datawidth = datawidth
        self.addrwidth = addrwidth
        self.mode = mode

        self.size = 1 << addrwidth

        self.ram = RAM(m, name + '_ram', clk, rst, datawidth, addrwidth, numports=2)

        self.front = m.Reg(name + '_front', addrwidth, signed=False, initval=0)
        if mode == 'rw':
            self.occupancy = tuple(m.Reg(name + '_occupancy_' + str(i), addrwidth + 1, signed=False, initval=0) for i in range(2))
        else:
            self.occupancy = m.Reg(name + '_occupancy', addrwidth + 1, signed=False, initval=0)
        self.base = m.Reg(name + '_base', addrwidth, signed=False, initval=0)
        if mode == 'wo':
            self.limit = m.Reg(name + '_limit', addrwidth + 2, signed=False, initval=self.size)
        else:
            self.limit = m.Reg(name + '_limit', addrwidth + 2, signed=False, initval=0)

        self.seq = Seq(m, name + '_seq', clk, rst)

        # synchronization
        if mode == 'rw':
            self.inc_occ = tuple(m.Wire(name + '_inc_occ_' + str(i), 1, signed=False) for i in range(2))
            self.dec_occ = tuple(m.Wire(name + '_dec_occ_' + str(i), 1, signed=False) for i in range(2))
            for i in range(2):
                self.seq.If(self.inc_occ[i], vtypes.Not(self.dec_occ[i]))(
                    self.occupancy[i].inc()
                ).Elif(vtypes.Not(self.inc_occ[i]), self.dec_occ[i])(
                    self.occupancy[i].dec()
                )
        else:
            self.inc_occ = m.Wire(name + '_inc_occ', 1, signed=False)
            self.dec_occ = m.Wire(name + '_dec_occ', 1, signed=False)
            self.seq.If(self.inc_occ, vtypes.Not(self.dec_occ))(
                self.occupancy.inc()
            ).Elif(vtypes.Not(self.inc_occ), self.dec_occ)(
                self.occupancy.dec()
            )

        # synchronization
        self.inc_lim = m.Wire(name + '_inc_lim', 1, signed=False)
        self.set_lim = m.Wire(name + '_set_lim', 1, signed=False)
        if mode == 'ro':
            self.seq.If(self.set_lim, self.inc_lim)(
                self.limit(self.occupancy + 1)
            ).Elif(self.set_lim)(
                self.limit(self.occupancy)
            ).Elif(self.inc_lim)(
                self.limit.inc()
            )
        elif mode == 'wo':
            self.seq.If(self.set_lim, self.inc_lim)(
                self.limit((self.size + 1) - self.occupancy)
            ).Elif(self.set_lim)(
                self.limit(self.size - self.occupancy)
            ).Elif(self.inc_lim)(
                self.limit.inc()
            )
        else:
            self.seq.If(self.set_lim, self.inc_lim)(
                # ternary addition possibly results in critical path
                self.limit(self.occupancy[1] - self.occupancy[0] + 1)
            ).Elif(self.set_lim)(
                self.limit(self.occupancy[1] - self.occupancy[0])
            ).Elif(self.inc_lim)(
                self.limit.inc()
            )

    @property
    def back(self):
        if self.mode == 'rw':
            return self.front + self.occupancy[1]
        else:
            return self.front + self.occupancy

    @property
    def vacancy(self):
        if self.mode == 'rw':
            return self.size - self.occupancy[1]
        else:
            return self.size - self.occupancy

    def dequeue(self, fsm: FSM) -> vtypes.Reg:
        if self.mode == 'ro':
            raise ValueError('Read-only mode does not support this operation')

        data, valid = self.ram.read_rtl(self.front, 1, fsm.here)
        data_reg = self.m.TmpReg(self.datawidth, signed=False, prefix='dequeue')
        fsm.If(valid)(
            data_reg(data)
        )

        self.seq.If(fsm.here, valid)(
            self.front.inc()
        )

        if self.mode == 'wo':
            self._add_cond(self.inc_lim, (fsm.here, valid))
            self._add_cond(self.dec_occ, (fsm.here, valid))
        else:
            for i in range(2):
                self._add_cond(self.dec_occ[i], (fsm.here, valid))
        fsm.If(valid).goto_next()

        return data_reg

    # used for DMA write and called by AXIM._synthesize_write_data_fsm_*
    # created based on RAM.read_burst
    # unused parameters: addr, stride, blocksize, rquit, port
    def dequeue_for_dma(self, addr, stride, length, blocksize, rready, rquit=False, port=0, cond=None) -> tuple[vtypes.Wire, vtypes.Reg, vtypes.Reg]:
        if self.mode == 'ro':
            raise ValueError('Read-only mode does not support this operation')

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='dequeue_fsm')
        length_reg = self.m.TmpReg(vtypes.get_width(length), prefix='dequeue_length')

        rvalid = self.m.TmpReg(1, signed=False, prefix='dequeue_rvalid')
        rlast = self.m.TmpReg(1, signed=False, prefix='dequeue_rlast')

        fsm(
            length_reg(length),
            rvalid(0),
            rlast(0)
        )
        fsm.If(cond, length > 0).goto_next()

        renable = vtypes.Ands(fsm.here, vtypes.Ors(vtypes.Not(rvalid), rready))
        rdata, _ = self.ram.read_rtl(self.front, 1, renable)
        rdata_wire = self.m.TmpWireLike(rdata, prefix='dequeue_rdata')
        rdata_wire.assign(rdata)

        self.seq.If(fsm.here, rready, length_reg > 0)(
            self.front.inc()
        )

        if self.mode == 'wo':
            self._add_cond(self.inc_lim, (fsm.here, rready, length_reg > 0))
            self._add_cond(self.dec_occ, (fsm.here, rready, length_reg > 0))
        else:
            for i in range(2):
                self._add_cond(self.dec_occ[i], (fsm.here, rready, length_reg > 0))

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

    def release(self, fsm: FSM) -> None:
        if self.mode == 'ro':
            self.seq.If(fsm.here)(
                self.front.inc()
            )
            self._add_cond(self.dec_occ, fsm.here)
        elif self.mode == 'wo':
            self._add_cond(self.inc_occ, fsm.here)
        else:
            self._add_cond(self.inc_occ[0], fsm.here)
        fsm.goto_next()

    def enqueue(self, fsm: FSM, data: vtypes.IntegralType) -> None:
        if self.mode == 'wo':
            raise ValueError('Write-only mode does not support this operation')

        self.ram.write_rtl(self.back, data, 1, fsm.here)

        self._add_cond(self.inc_lim, fsm.here)
        if self.mode == 'ro':
            self._add_cond(self.inc_occ, fsm.here)
        else:
            self._add_cond(self.inc_occ[1], fsm.here)
        fsm.goto_next()

    # used for DMA read and called by AXIM._synthesize_read_data_fsm_*
    # created based on RAM.write_burst
    # unused parameters: addr, stride, blocksize, wlast, wquit, port
    def enqueue_for_dma(self, addr, stride, length, blocksize, wdata, wvalid, wlast=False, wquit=False, port=0, cond=None) -> None:
        if self.mode == 'wo':
            raise ValueError('Write-only mode does not support this operation')

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='enqueue_for_dma')
        length_reg = self.m.TmpReg(vtypes.get_width(length), prefix='length')

        fsm(
            length_reg(length)
        )
        fsm.If(cond, length > 0).goto_next()

        self.ram.write_rtl(self.back, wdata, 1, (fsm.here, wvalid))

        self._add_cond(self.inc_lim, (fsm.here, wvalid))
        if self.mode == 'ro':
            self._add_cond(self.inc_occ, (fsm.here, wvalid))
        else:
            self._add_cond(self.inc_occ[1], (fsm.here, wvalid))

        fsm.If(wvalid)(
            length_reg.dec()
        )
        fsm.If(wvalid, length_reg <= 1).goto_init()

    def read(self, fsm: FSM, index: vtypes.IntegralType) -> vtypes.Reg:
        data, valid = self.ram.read_rtl(self.base + index, 0, (fsm.here, index < self.limit))
        data_reg = self.m.TmpReg(self.datawidth, signed=False, prefix='read')
        fsm.If(valid)(
            data_reg(data)
        )
        fsm.If(valid).goto_next()
        return data_reg

    def write(self, fsm: FSM, index: vtypes.IntegralType, data: vtypes.IntegralType) -> None:
        self.ram.write_rtl(self.base + index, data, 0, (fsm.here, index < self.limit))
        fsm.If(index < self.limit).goto_next()

    def rebase(self, fsm: FSM) -> None:
        self._add_cond(self.set_lim, fsm.here)
        if self.mode == 'ro':
            self.seq.If(fsm.here)(
                self.base(self.front)
            )
        elif self.mode == 'wo':
            self.seq.If(fsm.here)(
                self.base(self.front + self.occupancy)
            )
        else:
            self.seq.If(fsm.here)(
                self.base(self.front + self.occupancy[0])
            )
        fsm.goto_next()

    def _add_cond(self, tgt: vtypes._Variable, cond) -> None:
        cond = make_condition(cond)
        if tgt.assign_value is None:
            tgt.assign(cond)
        else:
            tgt.assign_value.statement.right = vtypes.Lor(cond, tgt.assign_value.statement.right)


__intrinsics__ = ('prefetch_dma_read', 'prefetch_dma_write')


def prefetch_dma_read(fsm: FSM):
    pass


def prefetch_dma_write(fsm: FSM):
    pass
