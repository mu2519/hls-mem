from veriloggen.core import module
from veriloggen.core import vtypes
from veriloggen.fsm.fsm import FSM, TmpFSM
from veriloggen.seq.seq import Seq
from .ram import RAM


class Buffet:
    __intrinsics__ = ('fill', 'read', 'update', 'shrink')

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
        self.clk = clk
        self.rst = rst
        self.datawidth = datawidth
        self.addrwidth = addrwidth
        self.ram = RAM(m, name + '_ram', clk, rst, datawidth, addrwidth, numports=2)
        self.head = m.Reg(name + '_head', addrwidth, signed=False, initval=0)
        self.occupancy = m.Reg(name + '_occupancy', addrwidth + 1, signed=False, initval=0)
        self.lock = m.Reg(name + '_lock', 1, signed=False, initval=0)
        self.seq = Seq(m, name + '_seq', clk, rst)

        self.size = 2**self.addrwidth

    def fill(self, fsm: FSM, data: vtypes.IntegralType) -> None:
        # implicit casting works as modulo operation
        addr = self.head + self.occupancy
        self.ram.write(fsm, addr, data, port=1)
        fsm(
            self.occupancy.inc()
        )
        fsm.goto_next()

    def fill_burst(self, addr, stride, length, blocksize, wdata, wvalid, wlast, wquit=False, port=0, cond=None):
        """
        Signature compatible with RAM.write_burst
        Invoking `fill` or `shrink` during invocation of `fill_burst` results in inconsistent states
        """

        # WARNING: the argument `addr` is ignored
        # WARNING: the argument `stride` is ignored and unit stride is used
        # WARNING: the argument `wquit` is ignored
        # WARNING: the argument `port` is ignored and port #1 is used

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='fill_burst_fsm')
        occupancy_reg = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='fill_burst_occupancy')
        length_reg = self.m.TmpReg(vtypes.get_width(length), initval=0, prefix='fill_burst_length')
        done = self.m.TmpReg(1, initval=0, prefix='fill_burst_done')

        # state 0
        fsm(
            length_reg(length),
            done(0)
        )
        fsm.If(cond, length > 0)(
            self.lock(1)
        )
        fsm.If(cond, length > 0).goto_next()

        # state 1
        fsm(
            occupancy_reg(self.occupancy)
        )
        fsm.goto_next()

        wenable = vtypes.Ands(fsm.here, wvalid)
        wready = fsm.here
        # implicit casting works as modulo operation
        addr = self.head + occupancy_reg
        self.ram.write_rtl(addr, wdata, port=1, cond=wenable)

        # state 2
        fsm(
            self.occupancy(occupancy_reg)
        )
        fsm.If(wvalid)(
            occupancy_reg.inc(),
            length_reg.dec()
        )
        fsm.If(vtypes.Ands(wvalid, vtypes.Ors(length_reg <= 1, wlast))).goto_next()

        # state 3
        fsm(
            self.occupancy(occupancy_reg),
            done(1),
            self.lock(0)
        )
        fsm.goto_init()

        return wready, done

    def read(self, fsm: FSM, index: vtypes.IntegralType) -> vtypes.Reg:
        # implicit casting works as modulo operation
        addr = self.head + index
        cond = self.occupancy > index
        return self.ram.read(fsm, addr, cond=cond, port=0)

    def update(self, fsm: FSM, index: vtypes.IntegralType, data: vtypes.IntegralType) -> None:
        """ for each index, read must be called at least once before update is called """

        # implicit casting works as modulo operation
        addr = self.head + index
        self.ram.write(fsm, addr, data, port=0)

    def shrink(self, fsm: FSM, num: vtypes.IntegralType) -> None:
        fsm.If(vtypes.Not(self.lock), self.occupancy >= num)(
            self.head.add(num),
            self.occupancy.sub(num)
        )
        fsm.Then().goto_next()

    def _id(self):
        return id(self)
