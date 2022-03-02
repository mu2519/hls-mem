from __future__ import absolute_import
from __future__ import print_function

import functools
import math

import veriloggen.core.vtypes as vtypes
import veriloggen.types.fixed as fxd
import veriloggen.types.util as util

from veriloggen.seq.seq import Seq, TmpSeq, make_condition
from veriloggen.fsm.fsm import TmpFSM
from veriloggen.types.ram import RAMInterface, mkRAMDefinition

from .ttypes import _MutexFunction


class RAM(_MutexFunction):
    __intrinsics__ = ('read', 'write') + _MutexFunction.__intrinsics__

    def __init__(self, m, name, clk, rst,
                 datawidth=32, addrwidth=10, numports=1,
                 initvals=None, nocheck_initvals=False,
                 ram_style=None, external_ports=None):

        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst

        self.datawidth = datawidth
        self.addrwidth = addrwidth
        self.numports = numports

        if external_ports is None:
            external_ports = ()

        self.interfaces = []

        for i in range(numports):
            if i in external_ports:
                interface = RAMInterface(m, name + '_%d' % i, datawidth, addrwidth,
                                         itype='Input', otype='Output', with_enable=True)
            else:
                interface = RAMInterface(m, name + '_%d' % i, datawidth, addrwidth,
                                         itype='Wire', otype='Wire', with_enable=True)

            self.interfaces.append(interface)

        for interface in self.interfaces:
            interface.wdata.no_write_check = True

        # default values
        for i, interface in enumerate(self.interfaces):
            if i not in external_ports:
                interface.addr.assign(vtypes.IntX())
                interface.wdata.assign(vtypes.IntX())
                interface.wenable.assign(0)
                interface.enable.assign(0)

        self.definition = mkRAMDefinition(
            name, datawidth, addrwidth, numports, initvals,
            with_enable=True,
            nocheck_initvals=nocheck_initvals,
            ram_style=ram_style)

        self.inst = self.m.Instance(self.definition, 'inst_' + name,
                                    ports=m.connect_ports(self.definition))

        self.seq = Seq(m, name, clk, rst)

        self.mutex = None

    def __getitem__(self, index):
        return self.interfaces[index]

    def _id(self):
        return id(self)

    @property
    def length(self):
        if isinstance(self.addrwidth, int):
            return 2 ** self.addrwidth
        return vtypes.Int(2) ** self.addrwidth

    def has_enable(self, port):
        return hasattr(self.interfaces[port], 'enable')

    def connect_rtl(self, port, addr, wdata=None, wenable=None, rdata=None, enable=None):
        """ connect native signals to the internal RAM interface """

        util.overwrite_assign(self.interfaces[port].addr, addr)
        if wdata is not None:
            util.overwrite_assign(self.interfaces[port].wdata, wdata)
        if wenable is not None:
            util.overwrite_assign(self.interfaces[port].wenable, wenable)
        if rdata is not None:
            rdata.connect(self.interfaces[port].rdata)

        if enable is not None:
            if self.has_enable(port):
                util.overwrite_assign(self.interfaces[port].enable, enable)
            else:
                raise ValueError("RAM '%s' has no enable port." % self.name)

        elif self.has_enable(port):
            raise ValueError('enable must be assigned.')

    def read_rtl(self, addr, port=0, cond=None):
        """
        @return data, valid
        """

        cond = make_condition(cond)

        if cond is not None:
            enable = cond
        else:
            enable = vtypes.Int(1, 1)

        util.add_mux(self.interfaces[port].addr, enable, addr)
        util.add_enable_cond(self.interfaces[port].enable, enable, vtypes.Int(1, 1))

        rdata = self.interfaces[port].rdata
        rvalid = self.seq.Prev(enable, 1)

        return rdata, rvalid

    def write_rtl(self, addr, wdata, port=0, cond=None):
        """
        @return None
        """
        cond = make_condition(cond)

        if cond is not None:
            enable = cond
        else:
            enable = vtypes.Int(1, 1)

        util.add_mux(self.interfaces[port].addr, enable, addr)
        util.add_mux(self.interfaces[port].wdata, enable, wdata)
        util.add_enable_cond(self.interfaces[port].wenable, enable, vtypes.Int(1, 1))
        util.add_enable_cond(self.interfaces[port].enable, enable, vtypes.Int(1, 1))

    def read(self, fsm, addr, port=0):
        """ intrinsic read operation using a shared Seq object """

        port = vtypes.to_int(port)
        cond = fsm.state == fsm.current

        rdata, rvalid = self.read_rtl(addr, port, cond)
        rdata_reg = self.m.TmpReg(self.datawidth, initval=0, signed=True)

        fsm.If(rvalid)(
            rdata_reg(rdata)
        )
        fsm.Then().goto_next()

        return rdata_reg

    def write(self, fsm, addr, wdata, port=0, cond=None):
        """ intrinsic write operation using a shared Seq object """

        port = vtypes.to_int(port)

        if cond is None:
            cond = fsm.state == fsm.current
        else:
            cond = vtypes.Ands(cond, fsm.state == fsm.current)

        self.write_rtl(addr, wdata, port, cond)
        fsm.goto_next()

        return 0

    def read_burst(self, addr, stride, length, rready, rquit=False, port=0, cond=None):
        """
        @return rdata, rvalid, rlast
        """

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='read_burst_fsm')

        _addr = self.m.TmpReg(self.addrwidth, initval=0, prefix='read_burst_addr')
        _stride = self.m.TmpReg(self.addrwidth, initval=0, prefix='read_burst_stride')
        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='read_burst_length')

        rvalid = self.m.TmpReg(prefix='read_burst_rvalid', initval=0)
        rlast = self.m.TmpReg(prefix='read_burst_rlast', initval=0)

        fsm(
            _addr(addr),
            _stride(stride),
            _length(length),
            rvalid(0),
            rlast(0),
        )
        fsm.If(cond, length > 0).goto_next()

        renable = vtypes.Ands(fsm.here, vtypes.Ors(vtypes.Not(rvalid), rready))
        rdata, _ = self.read_rtl(_addr, port, renable)
        rdata_wire = self.m.TmpWireLike(rdata, prefix='read_burst_rdata')
        rdata_wire.assign(rdata)

        fsm.If(rready, _length > 0)(
            _addr(_addr + _stride),
            _length.dec(),
            rvalid(1)
        )
        fsm.If(rready, _length <= 1)(
            rlast(1)
        )
        fsm.If(rlast, rvalid, rready)(
            rvalid(0),
            rlast(0),
        )
        fsm.If(rquit)(
            rvalid(0),
            rlast(0),
        )
        fsm.If(rlast, rvalid, rready).goto_init()
        fsm.If(rquit).goto_init()

        return rdata_wire, rvalid, rlast

    def write_burst(self, addr, stride, length, wdata, wvalid, wlast, wquit=False,
                    port=0, cond=None):
        """
        @return wready, done
        """

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='write_burst_fsm')

        _addr = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_addr')
        _stride = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_stride')
        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='write_burst_length')

        done = self.m.TmpReg(prefix='write_burst_done', initval=0)

        fsm(
            _addr(addr),
            _stride(stride),
            _length(length),
            done(0),
        )
        fsm.If(cond, length > 0).goto_next()

        wenable = vtypes.Ands(fsm.here, wvalid)
        wready = fsm.here
        self.write_rtl(_addr, wdata, port, wenable)

        fsm.If(wvalid)(
            _addr(_addr + _stride),
            _length.dec(),
            done(0)
        )
        fsm.If(wvalid, _length <= 1)(
            done(1)
        )
        fsm.If(wvalid, wlast)(
            done(1)
        )
        fsm.If(wvalid, _length <= 1).goto_init()
        fsm.If(wvalid, wlast).goto_init()
        fsm.If(wquit).goto_init()

        return wready, done


class FixedRAM(RAM):

    def __init__(self, m, name, clk, rst,
                 datawidth=32, addrwidth=10, numports=1, point=0,
                 initvals=None, nocheck_initvals=False, noconvert_initvals=False,
                 ram_style=None, external_ports=None):

        if initvals is not None and not noconvert_initvals:
            initvals = [fxd.to_fixed(initval, point) for initval in initvals]

        RAM.__init__(self, m, name, clk, rst,
                     datawidth, addrwidth, numports,
                     initvals, nocheck_initvals, ram_style, external_ports)

        self.point = point

    def read(self, fsm, addr, port=0, raw=False):
        raw_value = RAM.read(self, fsm, addr, port)
        if raw:
            return raw_value
        return fxd.reinterpret_cast_to_fixed(raw_value, self.point)

    def write(self, fsm, addr, wdata, port=0, cond=None, raw=False):
        if raw:
            fixed_wdata = wdata
        else:
            fixed_wdata = fxd.write_adjust(wdata, self.point)

        return RAM.write(self, fsm, addr, fixed_wdata, port, cond)


def extract_rams(rams):
    ret = []

    for ram in rams:
        if isinstance(ram, MultibankRAM):
            ret.extend(extract_rams(ram.rams))
        else:
            ret.append(ram)

    return ret


class MultibankRAM(object):
    __intrinsics__ = (
        'read', 'write',
        'read_bank', 'write_bank',
        'dma_read_bank', 'dma_read_bank_async',
        'dma_write_bank', 'dma_write_bank_async',
        'dma_read_block', 'dma_read_block_async',
        'dma_write_block', 'dma_write_block_async') + _MutexFunction.__intrinsics__

    def __init__(self, m, name, clk, rst,
                 datawidth=32, addrwidth=10, numports=1, numbanks=2,
                 ram_style=None, external_ports=None):

        if numbanks < 2:
            raise ValueError('numbanks must be 2 or more')

        self.m = m
        self.name = name
        self.clk = clk
        self.rst = rst
        self.orig_datawidth = datawidth
        self.datawidth = datawidth * numbanks
        self.addrwidth = addrwidth
        self.numports = numports
        self.numbanks = numbanks
        self.shift = util.log2(self.numbanks)
        self.rams = [RAM(m, '_'.join([name, '%d' % i]),
                         clk, rst, datawidth, addrwidth, numports,
                         ram_style=ram_style, external_ports=external_ports)
                     for i in range(numbanks)]
        self.keep_hierarchy = False
        self.seq = None

        # key: (axi._id(), port, ram_method_name)
        self.cache_dma_reqs = {}

        self.mutex = None

    def __getitem__(self, index):
        return self.rams[index]

    def _id(self):
        _ids = [ram._id() for ram in self.rams]
        return tuple(_ids)

    @property
    def length(self):
        if isinstance(self.addrwidth, int):
            return (2 ** self.addrwidth) * self.numbanks
        return (vtypes.Int(2) ** self.addrwidth) * self.numbanks

    def has_enable(self, port):
        for ram in self.rams:
            if not ram.has_enable(port):
                return False
        return True

    def connect_rtl(self, port, addr, wdata=None, wenable=None, rdata=None, enable=None):
        """ connect native signals to the internal RAM interface """

        if enable is not None:
            if not self.has_enable(port):
                raise ValueError("RAM '%s' has no enable port." % self.name)

        elif self.has_enable(port):
            raise ValueError('enable must be assigned.')

        if math.log(self.numbanks, 2) % 1.0 != 0.0:
            raise ValueError('numbanks must be power-of-2')

        if self.seq is None:
            self.seq = Seq(self.m, self.name, self.clk, self.rst)

        bank = self.m.TmpWire(self.shift)
        bank.assign(addr)
        addr = addr >> self.shift

        rdata_list = []
        for i, ram in enumerate(self.rams):
            if wenable is not None:
                bank_wenable = vtypes.Ands(wenable, bank == i)
            else:
                bank_wenable = None

            bank_rdata = self.m.TmpWire(self.orig_datawidth, signed=True)
            rdata_list.append(bank_rdata)

            if enable is not None:
                bank_enable = vtypes.Ands(enable, bank == i)
            else:
                bank_enable = None

            ram.connect_rtl(port, addr, wdata, bank_wenable, bank_rdata, bank_enable)

        bank_reg = self.seq.Prev(bank, 1, initval=0)
        pat = [(bank_reg == i, rdata_list[i])
               for i, ram in enumerate(self.rams)]
        pat.append((None, 0))

        rdata_wire = self.m.TmpWire(self.orig_datawidth, signed=True)
        rdata_wire.assign(vtypes.PatternMux(pat))

        if rdata is not None:
            rdata.connect(rdata_wire)

    def read_rtl(self, addr, port=0, cond=None):
        """
        @return data, valid
        """
        if math.log(self.numbanks, 2) % 1.0 != 0.0:
            raise ValueError('numbanks must be power-of-2')

        if self.seq is None:
            self.seq = Seq(self.m, self.name, self.clk, self.rst)

        rdata_list = []
        rvalid_list = []

        bank = self.m.TmpWire(self.shift)
        bank.assign(addr)
        addr = addr >> self.shift

        bank_reg = self.seq.Prev(bank, 1, initval=0)

        for ram in self.rams:
            rdata, rvalid = ram.read_rtl(addr, port, cond)
            rdata_list.append(rdata)
            rvalid_list.append(rvalid)

        rdata_wire = self.m.TmpWire(self.orig_datawidth, signed=True)
        rvalid_wire = self.m.TmpWire()

        pat = [(bank_reg == i, rdata_list[i])
               for i, ram in enumerate(self.rams)]
        pat.append((None, 0))

        rdata_wire.assign(vtypes.PatternMux(pat))
        rvalid_wire.assign(rvalid_list[0])

        return rdata_wire, rvalid_wire

    def write_rtl(self, addr, wdata, port=0, cond=None):
        """
        @return None
        """
        if math.log(self.numbanks, 2) % 1.0 != 0.0:
            raise ValueError('numbanks must be power-of-2')

        bank = self.m.TmpWire(self.shift)
        bank.assign(addr)
        addr = addr >> self.shift

        for i, ram in enumerate(self.rams):
            bank_cond = vtypes.Ands(cond, bank == i)
            ram.write_rtl(addr, wdata, port, bank_cond)

        return 0

    def _read_recursive(self, ram, port, addr, cond):
        if isinstance(ram, MultibankRAM):
            if math.log(ram.numbanks, 2) % 1.0 != 0.0:
                raise ValueError('numbanks must be power-of-2')

            rdata_list = []
            rvalid_list = []
            bank = self.m.TmpWire(ram.shift)
            bank.assign(addr)
            addr = addr >> ram.shift

            for sub in ram.rams:
                rdata, rvalid = self._read_recursive(sub, port, addr, cond)
                rdata_list.append(rdata)
                rvalid_list.append(rvalid)

            rdata_wire = self.m.TmpWire(ram.orig_datawidth, signed=True)

            patterns = [(bank == i, rdata)
                        for i, rdata in enumerate(rdata_list)]
            patterns.append((None, 0))
            rdata_wire.assign(vtypes.PatternMux(*patterns))

            return rdata_wire, rvalid_list[0]

        rdata, rvalid = ram.read_rtl(addr, port, cond)
        return rdata, rvalid

    def read(self, fsm, addr, port=0):
        if math.log(self.numbanks, 2) % 1.0 != 0.0:
            raise ValueError('numbanks must be power-of-2')

        port = vtypes.to_int(port)
        cond = fsm.state == fsm.current

        rdata_list = []
        rvalid_list = []

        bank = self.m.TmpWire(self.shift)
        bank.assign(addr)
        addr = addr >> self.shift

        for ram in self.rams:
            rdata, rvalid = self._read_recursive(ram, port, addr, cond)
            rdata_list.append(rdata)
            rvalid_list.append(rvalid)

        rdata_reg = self.m.TmpReg(self.orig_datawidth, initval=0, signed=True)

        for i, ram in enumerate(self.rams):
            fsm.If(rvalid_list[i], bank == i)(
                rdata_reg(rdata_list[i])
            )

        fsm.If(vtypes.Ors(*rvalid_list)).goto_next()

        return rdata_reg

    def _write_recursive(self, ram, port, addr, wdata, cond=None):
        if isinstance(ram, MultibankRAM):
            if math.log(ram.numbanks, 2) % 1.0 != 0.0:
                raise ValueError('numbanks must be power-of-2')

            bank = self.m.TmpWire(ram.shift)
            bank.assign(addr)
            addr = addr >> ram.shift

            for i, sub in enumerate(ram.rams):
                bank_cond = vtypes.Ands(cond, bank == i)
                self._write_recursive(sub, port, addr, wdata, bank_cond)

            return

        ram.write_rtl(addr, wdata, port, cond)

    def write(self, fsm, addr, wdata, port=0, cond=None):
        if math.log(self.numbanks, 2) % 1.0 != 0.0:
            raise ValueError('numbanks must be power-of-2')

        if cond is None:
            cond = fsm.state == fsm.current
        else:
            cond = vtypes.Ands(cond, fsm.state == fsm.current)

        bank = self.m.TmpWire(self.shift)
        bank.assign(addr)
        addr = addr >> self.shift

        for i, ram in enumerate(self.rams):
            bank_cond = vtypes.Ands(cond, bank == i)
            self._write_recursive(ram, port, addr, wdata, bank_cond)

        fsm.goto_next()

        return 0

    def read_bank(self, fsm, bank, addr, port=0):
        port = vtypes.to_int(port)
        cond = fsm.state == fsm.current

        rdata_list = []
        rvalid_list = []
        for ram in self.rams:
            rdata, rvalid = self._read_recursive(ram, port, addr, cond)
            rdata_list.append(rdata)
            rvalid_list.append(rvalid)

        rdata_reg = self.m.TmpReg(self.orig_datawidth, initval=0, signed=True)

        for i, ram in enumerate(self.rams):
            fsm.If(rvalid_list[i], bank == i)(
                rdata_reg(rdata_list[i])
            )

        fsm.If(vtypes.Ors(*rvalid_list)).goto_next()

        return rdata_reg

    def write_bank(self, fsm, bank, addr, wdata, port=0, cond=None):
        if cond is None:
            cond = fsm.state == fsm.current
        else:
            cond = vtypes.Ands(cond, fsm.state == fsm.current)

        for i, ram in enumerate(self.rams):
            bank_cond = vtypes.Ands(cond, bank == i)
            self._write_recursive(ram, port, addr, wdata, bank_cond)

        fsm.goto_next()

        return 0

    def dma_read_bank(self, fsm, bank, bus, local_addr, global_addr, size,
                      local_stride=1, port=0):

        self._dma_read_bank(fsm, bank, bus, local_addr, global_addr, size,
                            local_stride, port)

        bus.dma_wait_read(fsm)

    def dma_read_bank_async(self, fsm, bank, bus, local_addr, global_addr, size,
                            local_stride=1, port=0):

        self._dma_read_bank(fsm, bank, bus, local_addr, global_addr, size,
                            local_stride, port)

    def _dma_read_bank(self, fsm, bank, bus, local_addr, global_addr, size,
                       local_stride=1, port=0):
        check = fsm.current
        fsm.set_index(check + 1)

        starts = []
        ends = []
        for i, ram in enumerate(self.rams):
            starts.append(fsm.current)
            bus._dma_read(fsm, ram, local_addr, global_addr, size,
                          local_stride, port)
            ends.append(fsm.current)
            fsm.set_index(fsm.current + 1)

        fin = fsm.current

        for i, (s, e) in enumerate(zip(starts, ends)):
            fsm.goto_from(check, s, cond=bank == i)
            fsm.goto_from(e, fin)

    def dma_write_bank(self, fsm, bank, bus, local_addr, global_addr, size,
                       local_stride=1, port=0):

        self._dma_write_bank(fsm, bank, bus, local_addr, global_addr, size,
                             local_stride, port)

        bus.dma_wait_write(fsm)

    def dma_write_bank_async(self, fsm, bank, bus, local_addr, global_addr, size,
                             local_stride=1, port=0):

        self._dma_write_bank(fsm, bank, bus, local_addr, global_addr, size,
                             local_stride, port)

    def _dma_write_bank(self, fsm, bank, bus, local_addr, global_addr, size,
                        local_stride=1, port=0):
        check = fsm.current
        fsm.set_index(check + 1)

        starts = []
        ends = []
        for i, ram in enumerate(self.rams):
            starts.append(fsm.current)
            bus._dma_write(fsm, ram, local_addr, global_addr, size,
                           local_stride, port)
            ends.append(fsm.current)
            fsm.set_index(fsm.current + 1)

        fin = fsm.current

        for i, (s, e) in enumerate(zip(starts, ends)):
            fsm.goto_from(check, s, cond=bank == i)
            fsm.goto_from(e, fin)

    def dma_read_block(self, fsm, bus, local_addr, global_addr, size,
                       block_size=1, local_stride=1, port=0):

        self._dma_read_block(fsm, bus, local_addr, global_addr, size,
                             block_size, local_stride, port)

        bus.dma_wait_read(fsm)

    def dma_read_block_async(self, fsm, bus, local_addr, global_addr, size,
                             block_size=1, local_stride=1, port=0):

        self._dma_read_block(fsm, bus, local_addr, global_addr, size,
                             block_size, local_stride, port)

    def _dma_read_block(self, fsm, bus, local_addr, global_addr, size,
                        block_size=1, local_stride=1, port=0):

        cache_key = (id(bus), port)

        if cache_key in self.cache_dma_reqs:
            info = self.cache_dma_reqs[cache_key]
            seq = info[0]
            req_block_size = info[1]
        else:
            seq = TmpSeq(bus.m, bus.clk, bus.rst)
            req_block_size = self.m.TmpReg(self.addrwidth, initval=0,
                                           prefix='req_block_size')
            info = (seq, req_block_size)
            self.cache_dma_reqs[cache_key] = info

        set_req = bus._set_flag(fsm, prefix='set_req')
        seq.If(set_req)(
            req_block_size(block_size)
        )

        ram_method = functools.partial(self.write_burst_block,
                                       blocksize=req_block_size)

        bus._dma_read(fsm, self, local_addr, global_addr, size,
                      local_stride, port, ram_method)

    def dma_write_block(self, fsm, bus, local_addr, global_addr, size,
                        block_size=1, local_stride=1, port=0):

        self._dma_write_block(fsm, bus, local_addr, global_addr, size,
                              block_size, local_stride, port)

        bus.dma_wait_write(fsm)

    def dma_write_block_async(self, fsm, bus, local_addr, global_addr, size,
                              block_size=1, local_stride=1, port=0):

        self._dma_write_block(fsm, bus, local_addr, global_addr, size,
                              block_size, local_stride, port)

    def _dma_write_block(self, fsm, bus, local_addr, global_addr, size,
                         block_size=1, local_stride=1, port=0):

        cache_key = (id(bus), port)

        if cache_key in self.cache_dma_reqs:
            info = self.cache_dma_reqs[cache_key]
            seq = info[0]
            req_block_size = info[1]
        else:
            seq = TmpSeq(bus.m, bus.clk, bus.rst)
            req_block_size = self.m.TmpReg(self.addrwidth, initval=0,
                                           prefix='req_block_size')
            info = (seq, req_block_size)
            self.cache_dma_reqs[cache_key] = info

        set_req = bus._set_flag(fsm, prefix='set_req')
        seq.If(set_req)(
            req_block_size(block_size)
        )

        ram_method = functools.partial(self.read_burst_block,
                                       blocksize=req_block_size)

        bus._dma_write(fsm, self, local_addr, global_addr, size,
                       local_stride, port, ram_method)

    def read_burst(self, addr, stride, length, rready, rquit=False, port=0, cond=None):
        """
        @return rdata, rvalid, rlast
        """

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='read_burst_fsm')

        _addr = self.m.TmpReg(self.addrwidth, initval=0, prefix='read_burst_addr')
        _stride = self.m.TmpReg(self.addrwidth, initval=0, prefix='read_burst_stride')
        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='read_burst_length')

        rvalid = self.m.TmpReg(prefix='read_burst_rvalid', initval=0)
        rlast = self.m.TmpReg(prefix='read_burst_rlast', initval=0)

        fsm(
            _addr(addr),
            _stride(stride),
            _length(length),
            rvalid(0),
            rlast(0),
        )
        fsm.If(cond, length > 0).goto_next()

        renable = vtypes.Ands(fsm.here, vtypes.Ors(vtypes.Not(rvalid), rready))

        ram_rdata_wires = []
        for ram in self.rams:
            rdata, _ = ram.read_rtl(_addr, port, renable)
            ram_rdata_wire = self.m.TmpWireLike(rdata, prefix='read_burst_ram_rdata')
            ram_rdata_wire.assign(rdata)
            ram_rdata_wires.append(ram_rdata_wire)

        ram_rdata_wires.reverse()
        rdata_wire = self.m.TmpWire(self.datawidth, prefix='read_burst_rdata')
        rdata_wire.assign(vtypes.Cat(*ram_rdata_wires))

        fsm.If(rready, _length > 0)(
            _addr(_addr + _stride),
            _length.dec(),
            rvalid(1)
        )
        fsm.If(rready, _length <= 1)(
            rlast(1)
        )
        fsm.If(rlast, rvalid, rready)(
            rvalid(0),
            rlast(0),
        )
        fsm.If(rquit)(
            rvalid(0),
            rlast(0),
        )
        fsm.If(rlast, rvalid, rready).goto_init()
        fsm.If(rquit).goto_init()

        return rdata_wire, rvalid, rlast

    def read_burst_block(self, addr, stride, length, rready, rquit=False,
                         port=0, cond=None, blocksize=1):
        """
        @return rdata, rvalid, rlast
        """

        ram_sel_list = []
        ram_rready_list = []
        ram_rquit_list = []
        ram_rdata_list = []

        for ram in self.rams:
            ram_sel = self.m.TmpReg(prefix='read_burst_block_ram_sel', initval=0)
            ram_rready = self.m.TmpWire(prefix='read_burst_block_ram_rready')
            ram_rquit = self.m.TmpWire(prefix='read_burst_block_ram_rquit')
            ram_rdata, _, _ = ram.read_burst(addr, stride, length, ram_rready, ram_rquit, port, cond)
            ram_sel_list.append(ram_sel)
            ram_rready_list.append(ram_rready)
            ram_rquit_list.append(ram_rquit)
            ram_rdata_list.append(ram_rdata)

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='read_burst_block_fsm')

        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='read_burst_block_length')
        _blocksize = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_block_blocksize')
        count = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_block_count')
        rvalid = self.m.TmpReg(prefix='read_burst_block_rvalid', initval=0)
        rlast = self.m.TmpReg(prefix='read_burst_block_rlast', initval=0)

        fsm(
            _length(length),
            _blocksize(blocksize),
            count(0),
            rvalid(0),
            rlast(0),
        )

        for ram_sel in ram_sel_list:
            fsm(
                ram_sel(0)
            )

        fsm.If(cond, length > 0).goto_next()

        rdata = self.m.TmpWire(self.rams[0].datawidth,
                               prefix='read_burst_block_rdata')
        loop = fsm.current

        for i, (ram_rready, ram_rquit, ram_rdata) in enumerate(
                zip(ram_rready_list, ram_rquit_list, ram_rdata_list)):

            ram_rready.assign(vtypes.Ands(rready, fsm.here))
            ram_rquit.assign(vtypes.Ors(rquit, vtypes.Ands(rvalid, rlast)))

            util.add_mux(rdata, ram_sel_list[i], ram_rdata)
            for j, ram_sel in enumerate(ram_sel_list):
                if i == j:
                    fsm.If(rready)(
                        ram_sel(1)
                    )
                else:
                    fsm.If(rready)(
                        ram_sel(0)
                    )

            fsm.If(rready, _length > 0)(
                _length.dec(),
                count.inc(),
                rvalid(1)
            )
            fsm.If(rready, _length <= 1)(
                rlast(1)
            )
            fsm.If(rlast, rvalid, rready)(
                rvalid(0),
                rlast(0),
            )
            fsm.If(rquit)(
                rvalid(0),
                rlast(0),
            )
            fsm.If(rready, count == _blocksize - 1)(
                count(0)
            )

            if i == len(ram_rready_list) - 1:
                fsm.If(rready, count == _blocksize - 1).goto(loop)
            else:
                fsm.If(rready, count == _blocksize - 1).goto(fsm.next)

            fsm.If(rlast, rvalid, rready).goto_init()
            fsm.If(rquit).goto_init()

            fsm.inc()

        return rdata, rvalid, rlast

    def write_burst(self, addr, stride, length, wdata, wvalid, wlast, wquit=False,
                    port=0, cond=None):
        """
        @return wready, done
        """

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='write_burst_fsm')

        _addr = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_addr')
        _stride = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_stride')
        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='write_burst_length')

        done = self.m.TmpReg(prefix='write_burst_done', initval=0)

        fsm(
            _addr(addr),
            _stride(stride),
            _length(length),
            done(0),
        )
        fsm.If(cond, length > 0).goto_next()

        wenable = vtypes.Ands(fsm.here, wvalid)
        wready = fsm.here
        for i, ram in enumerate(self.rams):
            ram_wdata = self.m.TmpWire(ram.datawidth, prefix='write_burst_ram_wdata')
            ram_wdata.assign(vtypes.Srl(wdata, self.orig_datawidth * i))
            ram.write_rtl(_addr, ram_wdata, port, wenable)

        fsm.If(wvalid)(
            _addr(_addr + _stride),
            _length.dec(),
            done(0)
        )
        fsm.If(wvalid, _length <= 1)(
            done(1)
        )
        fsm.If(wvalid, wlast)(
            done(1)
        )
        fsm.If(wvalid, _length <= 1).goto_init()
        fsm.If(wvalid, wlast).goto_init()
        fsm.If(wquit).goto_init()

        return wready, done

    def write_burst_bcast(self, addr, stride, length, wdata, wvalid, wlast, wquit=False,
                          port=0, cond=None):
        """
        @return wready, done
        """
        wready_list = []
        done_list = []

        for ram in self.rams:
            wready, done = ram.write_burst(addr, stride, length, wdata, wvalid, wlast, wquit,
                                           port, cond)
            wready_list.append(wready)
            done_list.append(done_list)

        return wready_list[0], done_list[0]

    def write_burst_block(self, addr, stride, length, wdata, wvalid, wlast,
                          wquit=False, port=0, cond=None, blocksize=1):
        """
        @return wready, done
        """

        ram_wvalid_list = []
        ram_wquit_list = []

        for ram in self.rams:
            ram_wvalid = self.m.TmpWire(prefix='write_burst_block_ram_wvalid')
            ram_wquit = self.m.TmpWire(prefix='write_burst_block_ram_wquit')
            _ = ram.write_burst(addr, stride, length, wdata, ram_wvalid, wlast,
                                ram_wquit, port, cond)
            ram_wvalid_list.append(ram_wvalid)
            ram_wquit_list.append(ram_wquit)

        fsm = TmpFSM(self.m, self.clk, self.rst, prefix='write_burst_block_fsm')

        _length = self.m.TmpReg(self.addrwidth + 1, initval=0, prefix='write_burst_block_length')
        _blocksize = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_block_blocksize')
        done = self.m.TmpReg(prefix='write_burst_block_done', initval=0)
        count = self.m.TmpReg(self.addrwidth, initval=0, prefix='write_burst_block_count')

        fsm(
            _length(length),
            _blocksize(blocksize),
            done(0),
            count(0),
        )
        fsm.If(cond, length > 0).goto_next()

        loop = fsm.current
        wready = False

        for i, (ram_wvalid, ram_wquit) in enumerate(zip(ram_wvalid_list, ram_wquit_list)):

            ram_wvalid.assign(vtypes.Ands(wvalid, fsm.here))
            ram_wquit.assign(vtypes.Ors(wquit, vtypes.Ands(wvalid, wlast), vtypes.Ands(wvalid, _length <= 1)))

            fsm.If(wvalid)(
                _length.dec(),
                done(0),
                count.inc(),
            )
            fsm.If(wvalid, _length <= 1)(
                done(1)
            )
            fsm.If(wvalid, wlast)(
                done(1)
            )
            fsm.If(wvalid, count == _blocksize - 1)(
                count(0)
            )

            if i == len(ram_wvalid_list) - 1:
                fsm.If(wvalid, count == _blocksize - 1).goto(loop)
            else:
                fsm.If(wvalid, count == _blocksize - 1).goto(fsm.next)

            fsm.If(wvalid, _length <= 1).goto_init()
            fsm.If(wvalid, wlast).goto_init()
            fsm.If(wquit).goto_init()

            fsm.inc()

            wready = vtypes.Ors(wready, fsm.here)

        return wready, done


class _PackedMultibankRAM(MultibankRAM):
    def __init__(self, src=None, name=None, keep_hierarchy=False):

        if not isinstance(src, (tuple, list)):
            src = [src]

        if not keep_hierarchy:
            src = extract_rams(src)

        if len(src) < 2:
            raise ValueError('numbanks must be 2 or more')

        max_datawidth = 0
        for ram in src:
            max_datawidth = max(max_datawidth, ram.datawidth)

        max_addrwidth = 0
        for ram in src:
            max_addrwidth = max(max_addrwidth, ram.addrwidth)

        max_numports = src[0].numports
        for ram in src[1:]:
            if max_numports != ram.numports:
                raise ValueError('numports must be same')

        self.m = src[0].m
        self.name = ('_'.join([ram.name for ram in src])
                     if name is None else name)
        self.clk = src[0].clk
        self.rst = src[0].rst
        self.orig_datawidth = max_datawidth
        self.datawidth = max_datawidth * len(src)
        self.addrwidth = max_addrwidth
        self.numports = max_numports
        self.numbanks = len(src)
        self.shift = util.log2(self.numbanks)
        self.rams = src
        self.keep_hierarchy = keep_hierarchy
        self.seq = None

        for ram in self.rams:
            if ram.seq is not None:
                self.seq = ram.seq
                break

        # key: (axi._id(), port, ram_method_name)
        self.cache_dma_reqs = {}

        self.mutex = None


multibank_ram_cache = {}


def to_multibank_ram(rams, name=None, keep_hierarchy=False):
    ids = tuple([ram._id() for ram in rams])

    if ids in multibank_ram_cache:
        return multibank_ram_cache[ids]

    ram = _PackedMultibankRAM(rams, name, keep_hierarchy)
    multibank_ram_cache[ids] = ram

    return ram
