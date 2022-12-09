from __future__ import annotations

import copy
import ast
from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal, Any

from veriloggen.core import vtypes
from veriloggen.types import fixed


RegLike = vtypes.Reg | fixed._FixedReg


class ScopeName(object):

    def __init__(self, namelist: Sequence[str]):
        self.namelist = namelist

    def __repr__(self):
        return '_'.join(self.namelist)


class ScopeFrameList(object):
    """
    ScopeFrameList is similar to a call stack
    and not necessarily corresponds to the notion of scopes
    """

    def __init__(self):
        self.current = ScopeFrame(ScopeName(('_',)))
        self.previousframes: OrderedDict[ScopeFrame, ScopeFrame | None] = OrderedDict()
        self.previousframes[self.current] = None
        self.nextframes: OrderedDict[ScopeFrame, list[ScopeFrame]] = OrderedDict()
        self.label_prefix = 's'
        self.label_count = 0

    def getNamePrefix(self):
        return self.current.getNamePrefix()

    def getCurrent(self):
        return self.current

    def pushScopeFrame(self, name: str | None = None, ftype: Literal['call'] | None = None):
        if name is None:
            name = self.label_prefix + str(self.label_count)
            self.label_count += 1
        prefix = copy.deepcopy(self.current.name.namelist)
        framename = ScopeName(prefix + (name,))
        f = ScopeFrame(framename, ftype)
        self.previousframes[f] = self.current
        if self.current not in self.nextframes:
            self.nextframes[self.current] = []
        self.nextframes[self.current].append(f)
        self.current = f

    def popScopeFrame(self):
        self.current = self.previousframes[self.current]

    def addVariable(self, name: str, var: RegLike):
        if self.current is None:
            return None
        targ = self.current
        while targ is not None:
            if targ.ftype == 'call':
                targ.addVariable(name, var)
                break
            targ = self.previousframes[targ]
        return None

    def addFunction(self, func: ast.FunctionDef):
        self.current.addFunction(func)

    def searchVariable(self, name: str):
        if self.current is None:
            return None

        targ = self.current

        while targ is not None:
            ret = targ.searchVariable(name)
            if ret is not None:
                return ret

            ret = targ.searchFunction(name)
            if ret is not None:
                return ret

            if targ.ftype == 'call':
                break

            targ = self.previousframes[targ]

        return None

    def searchFunction(self, name: str):
        if self.current is None:
            return None

        targ = self.current

        while targ is not None:
            ret = targ.searchFunction(name)
            if ret is not None:
                return ret

            targ = self.previousframes[targ]

        return None

    def addBreak(self, count):
        self.current.addBreak(count)

    def addContinue(self, count):
        self.current.addContinue(count)

    def addReturn(self, count, value):
        self.current.addReturn(count, value)

    def setReturnVariable(self, var: RegLike):
        if self.current is None:
            return
        targ = self.current
        while targ is not None:
            if targ.ftype == 'call':
                targ.setReturnVariable(var)
                return
            targ = self.previousframes[targ]

    def hasBreak(self):
        return self.current.hasBreak()

    def hasContinue(self):
        return self.current.hasContinue()

    def hasReturn(self):
        return self.current.hasReturn()

    def getUnresolvedBreak(self, p: ScopeFrame | None = None):
        ret: list[int] = []
        ptr = self.current if p is None else p
        ret.extend(ptr.getBreak())
        if ptr not in self.nextframes:
            return tuple(ret)
        for f in self.nextframes[ptr]:
            ret.extend(self.getUnresolvedBreak(f))
        return tuple(ret)

    def getUnresolvedContinue(self, p: ScopeFrame | None = None):
        ret: list[int] = []
        ptr = self.current if p is None else p
        ret.extend(ptr.getContinue())
        if ptr not in self.nextframes:
            return tuple(ret)
        for f in self.nextframes[ptr]:
            ret.extend(self.getUnresolvedContinue(f))
        return tuple(ret)

    def getUnresolvedReturn(self, p: ScopeFrame | None = None):
        ret: list[tuple[int, Any]] = []
        ptr = self.current if p is None else p
        ret.extend(ptr.getReturn())
        if ptr not in self.nextframes:
            return tuple(ret)
        for f in self.nextframes[ptr]:
            ret.extend(self.getUnresolvedReturn(f))
        return tuple(ret)

    def getReturnVariable(self):
        if self.current is None:
            return None
        targ = self.current
        while targ is not None:
            if targ.ftype == 'call':
                return targ.getReturnVariable()
            targ = self.previousframes[targ]
        return None

    def clearBreak(self, p: ScopeFrame | None = None):
        ptr = self.current if p is None else p
        ptr.clearBreak()
        if ptr not in self.nextframes:
            return
        for f in self.nextframes[ptr]:
            self.clearBreak(f)

    def clearContinue(self, p: ScopeFrame | None = None):
        ptr = self.current if p is None else p
        ptr.clearContinue()
        if ptr not in self.nextframes:
            return
        for f in self.nextframes[ptr]:
            self.clearContinue(f)

    def clearReturn(self, p: ScopeFrame | None = None):
        ptr = self.current if p is None else p
        ptr.clearReturn()
        if ptr not in self.nextframes:
            return
        for f in self.nextframes[ptr]:
            self.clearReturn(f)

    def clearReturnVariable(self, p: ScopeFrame | None = None):
        ptr = self.current if p is None else p
        ptr.clearReturnVariable()
        if ptr not in self.nextframes:
            return
        for f in self.nextframes[ptr]:
            self.clearReturnVariable(f)


class ScopeFrame(object):
    """
    ScopeFrame is actually similar to a stack frame
    and not necessarily corresponds to the notion of scopes
    """

    def __init__(self, name: ScopeName, ftype: Literal['call'] | None = None):
        self.name = name
        self.ftype = ftype
        self.variables: OrderedDict[str, RegLike] = OrderedDict()
        self.functions: OrderedDict[str, ast.FunctionDef] = OrderedDict()
        self.unresolved_break: list[int] = []
        self.unresolved_continue: list[int] = []
        self.unresolved_return: list[tuple[int, Any]] = []
        self.returnvariable: RegLike | None = None

    def getNamePrefix(self):
        return str(self.name)

    def addVariable(self, name: str, var: RegLike):
        self.variables[name] = var

    def addFunction(self, func: ast.FunctionDef):
        name = func.name
        self.functions[name] = func

    def searchVariable(self, name: str):
        if name not in self.variables:
            return None
        return self.variables[name]

    def searchFunction(self, name: str):
        if name not in self.functions:
            return None
        return self.functions[name]

    def addBreak(self, count: int):
        self.unresolved_break.append(count)

    def addContinue(self, count: int):
        self.unresolved_continue.append(count)

    def addReturn(self, count: int, value: Any):
        self.unresolved_return.append((count, value))

    def hasBreak(self):
        if self.unresolved_break:
            return True
        return False

    def hasContinue(self):
        if self.unresolved_continue:
            return True
        return False

    def hasReturn(self):
        if self.unresolved_return:
            return True
        return False

    def getBreak(self):
        return self.unresolved_break

    def getContinue(self):
        return self.unresolved_continue

    def getReturn(self):
        return self.unresolved_return

    def clearBreak(self):
        self.unresolved_break = []

    def clearContinue(self):
        self.unresolved_continue = []

    def clearReturn(self):
        self.unresolved_return = []

    def clearReturnVariable(self):
        self.returnvariable = None

    def setReturnVariable(self, var: RegLike):
        self.returnvariable = var

    def getReturnVariable(self):
        return self.returnvariable

    def __hash__(self):
        return hash((self.name, self.ftype, id(self)))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        if self.ftype != other.ftype:
            return False
        if id(self) != id(other):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
