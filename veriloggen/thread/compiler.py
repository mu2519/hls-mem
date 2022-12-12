from __future__ import annotations

from functools import partial, reduce
import copy
import ast
import inspect
import textwrap
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from typing import Literal, Any, Callable
from types import FunctionType, MethodType, FrameType

from veriloggen.fsm.fsm import FSM
from veriloggen.core.module import Module
import veriloggen.core.vtypes as vtypes
import veriloggen.types.fixed as fxd
from veriloggen.optimizer import try_optimize as optimize
from .scope import ScopeFrameList
from .operator import getVeriloggenOp, getMethodName, applyMethod
from .fixed import FixedConst
from .pipo import PIPO
from .inchworm import Inchworm

numerical_types = vtypes.numerical_types

_tmp_count = 0


# compiler.py: Python AST -> FSM


def union(x: set, y: set) -> set:
    return x | y


def get_vars(code: ast.AST | Sequence[ast.AST], ctx: Literal['load', 'store', 'both'] = 'both') -> set[str]:
    """
    extract variables which have the specified context
    `ctx`: the context of variables
    """
    match code:
        case [*_] as nodes:
            return reduce(union, map(partial(get_vars, ctx=ctx), nodes), set())
        case ast.AST() as node:
            match ctx:
                case 'load':
                    if isinstance(node, ast.AugAssign):
                        return get_vars(node.target, 'store') | get_vars(node.value, 'load')
                    if isinstance(node, ast.Name):
                        return {node.id} if isinstance(node.ctx, ast.Load) else set()
                case 'store':
                    if isinstance(node, ast.Name):
                        return {node.id} if isinstance(node.ctx, ast.Store) else set()
                case 'both':
                    if isinstance(node, ast.Name):
                        return {node.id}
                case _:
                    raise ValueError
            return reduce(union, map(partial(get_vars, ctx=ctx), ast.iter_child_nodes(node)), set())
        case _:
            raise TypeError


def find_dma(code: ast.AST | Sequence[ast.AST], ram_name: str) -> bool:
    """
    judge whether DMA accesses tied to the specified RAM object exist
    `ram_name`: the identifier of the RAM object
    """
    if isinstance(code, ast.AST):
        node = code
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            attr = node.func.attr
            if isinstance(value, ast.Name) and value.id == ram_name:
                if attr in ['dma_read', 'dma_write', 'push']:
                    return True
        for n in ast.iter_child_nodes(node):
            if find_dma(n, ram_name):
                return True
    else:
        nodes = code
        for n in nodes:
            if find_dma(n, ram_name):
                return True
    return False


def get_dma_vars(code: ast.AST | Sequence[ast.AST], ram_name: str) -> set[str]:
    """
    extract variables which occur in the arguments of function calls
    for DMA accesses tied to the specified RAM object
    `ram_name`: the identifier of the RAM object
    """
    if isinstance(code, ast.AST):
        node = code
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == ram_name:
                if node.func.attr in ['dma_read', 'dma_write', 'push']:
                    return get_vars(node.args, 'both') | get_vars([kw.value for kw in node.keywords], 'both')
        return reduce(union, map(partial(get_dma_vars, ram_name=ram_name), ast.iter_child_nodes(node)), set())
    else:
        nodes = code
        return reduce(union, map(partial(get_dma_vars, ram_name=ram_name), nodes), set())


def make_var_dep_graph_sub(node: ast.AST, rslt: defaultdict[str, set[str]]) -> None:
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        if isinstance(node, ast.Assign):
            dst_vars = get_vars(node.targets)
        else:
            dst_vars = get_vars(node.target)
        src_vars = get_vars(node.value)
        for v in dst_vars:
            rslt[v] |= src_vars
    else:
        for n in ast.iter_child_nodes(node):
            make_var_dep_graph_sub(n, rslt)


def make_var_dep_graph(node: ast.AST) -> defaultdict[str, set[str]]:
    """ construct a variable dependency graph from an AST """
    rslt: defaultdict[str, set[str]] = defaultdict(set)
    make_var_dep_graph_sub(node, rslt)
    return rslt


def collect_reachable_sub(graph: dict[Any, set], node: Any, visited: set) -> None:
    if node in visited:
        return
    visited.add(node)
    for n in graph[node]:
        collect_reachable_sub(graph, n, visited)


def collect_reachable(graph: dict[Any, set], start_nodes: set) -> set:
    """ collect reachable nodes given a graph and a set of start nodes """
    visited = set()
    for s in start_nodes:
        collect_reachable_sub(graph, s, visited)
    return visited


def extract_dma_relevant_sub(node: ast.stmt, ram: str, needed_vars: set[str]) -> ast.stmt | None:
    if isinstance(node, ast.For) and node.orelse:
        raise ValueError('for-else statement is not supported')
    if isinstance(node, ast.While) and node.orelse:
        raise ValueError('while-else statement is not supported')

    if isinstance(node, (ast.FunctionDef, ast.For, ast.While)):
        body = map(partial(extract_dma_relevant_sub, ram=ram, needed_vars=needed_vars), node.body)
        body = filter(lambda x: x is not None, body)
        body = list(body)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            return node
        else:
            return None
    elif isinstance(node, ast.Assign):
        if get_vars(node.targets) & needed_vars:
            return node
    elif isinstance(node, ast.AugAssign):
        if get_vars(node.target) & needed_vars:
            return node
    elif isinstance(node, ast.If):
        body = map(partial(extract_dma_relevant_sub, ram=ram, needed_vars=needed_vars), node.body)
        orelse = map(partial(extract_dma_relevant_sub, ram=ram, needed_vars=needed_vars), node.orelse)
        body = filter(lambda x: x is not None, body)
        orelse = filter(lambda x: x is not None, orelse)
        body = list(body)
        orelse = list(orelse)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            node.orelse = orelse
            return node
        elif orelse:
            node = copy.deepcopy(node)
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            node.body = orelse
            node.orelse = []
            return node
        else:
            return None
    elif isinstance(node, ast.Pass):
        return None
    elif isinstance(node, (ast.Return, ast.Break, ast.Continue)):
        return node

    if find_dma(node, ram):
        return node
    else:
        return None


def extract_dma_relevant(node: ast.stmt, ram: str) -> ast.stmt | None:
    var_dep_graph = make_var_dep_graph(node)
    needed_vars = collect_reachable(var_dep_graph, get_dma_vars(node, ram))
    return extract_dma_relevant_sub(node, ram, needed_vars)


def find_side_effect(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        return True
    return any(map(find_side_effect, ast.iter_child_nodes(node)))


def get_side_effect_vars(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call):
        return get_vars(node.args) | get_vars([kw.value for kw in node.keywords])
    return reduce(union, map(get_side_effect_vars, ast.iter_child_nodes(node)), set())


def extract_dma_irrelevant_sub(node: ast.stmt, needed_vars: set[str]) -> ast.stmt | None:
    if isinstance(node, ast.For) and node.orelse:
        raise ValueError('for-else statement is not supported')
    if isinstance(node, ast.While) and node.orelse:
        raise ValueError('while-else statement is not supported')

    if isinstance(node, (ast.FunctionDef, ast.For, ast.While)):
        body = map(partial(extract_dma_irrelevant_sub, needed_vars=needed_vars), node.body)
        body = filter(lambda x: x is not None, body)
        body = list(body)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            return node
        else:
            return None
    elif isinstance(node, ast.Assign):
        if get_vars(node.targets) & needed_vars:
            return node
    elif isinstance(node, ast.AugAssign):
        if get_vars(node.target) & needed_vars:
            return node
    elif isinstance(node, ast.If):
        body = map(partial(extract_dma_irrelevant_sub, needed_vars=needed_vars), node.body)
        orelse = map(partial(extract_dma_irrelevant_sub, needed_vars=needed_vars), node.orelse)
        body = filter(lambda x: x is not None, body)
        orelse = filter(lambda x: x is not None, orelse)
        body = list(body)
        orelse = list(orelse)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            node.orelse = orelse
            return node
        elif orelse:
            node = copy.deepcopy(node)
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            node.body = orelse
            node.orelse = []
            return node
        else:
            return None
    elif isinstance(node, ast.Pass):
        return None
    elif isinstance(node, (ast.Return, ast.Break, ast.Continue)):
        return node

    if find_side_effect(node):
        return node
    else:
        return None


def eliminate_dma_calls(node: ast.stmt, rams: list[str]) -> ast.stmt | None:
    if isinstance(node, ast.For) and node.orelse:
        raise ValueError('for-else statement is not supported')
    if isinstance(node, ast.While) and node.orelse:
        raise ValueError('while-else statement is not supported')

    if isinstance(node, (ast.FunctionDef, ast.For, ast.While)):
        body = map(partial(eliminate_dma_calls, rams=rams), node.body)
        body = filter(lambda x: x is not None, body)
        body = list(body)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            return node
        else:
            return None
    elif isinstance(node, ast.If):
        body = map(partial(eliminate_dma_calls, rams=rams), node.body)
        orelse = map(partial(eliminate_dma_calls, rams=rams), node.orelse)
        body = filter(lambda x: x is not None, body)
        orelse = filter(lambda x: x is not None, orelse)
        body = list(body)
        orelse = list(orelse)
        if body:
            node = copy.deepcopy(node)
            node.body = body
            node.orelse = orelse
            return node
        elif orelse:
            node = copy.deepcopy(node)
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            node.body = orelse
            node.orelse = []
            return node
        else:
            return None
    elif isinstance(node, ast.Pass):
        return None
    elif isinstance(node, (ast.Return, ast.Break, ast.Continue)):
        return node

    for r in rams:
        if find_dma(node, r):
            return None
    return node


def extract_dma_irrelevant(node: ast.stmt, rams: list[str]) -> ast.stmt | None:
    node = eliminate_dma_calls(node, rams)
    if node is None:
        return None
    var_dep_graph = make_var_dep_graph(node)
    needed_vars = collect_reachable(var_dep_graph, get_side_effect_vars(node))
    return extract_dma_irrelevant_sub(node, needed_vars)


def _tmp_name(prefix='_tmp_thread'):
    global _tmp_count
    v = _tmp_count
    _tmp_count += 1
    ret = '_'.join([prefix, str(v)])
    return ret


class CompileError(Exception):

    def __init__(self, err, code, lineno, col_offset,
                 end_lineno=None, end_col_offset=None):
        self.err = err
        self.code = code
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset

    def __str__(self):
        return '"{}" in "{}", line {}'.format(self.err, self.code, self.lineno)


class FunctionVisitor(ast.NodeVisitor):

    def __init__(self):
        self.functions: OrderedDict[str, ast.FunctionDef] = OrderedDict()

    def visit(self, node):
        try:
            r = super().visit(node)
            return r

        except CompileError:
            raise

        except Exception as e:
            # ast.unparse was added in Python 3.9
            if hasattr(ast, 'unparse'):  # Python 3.9 or later
                code = ast.unparse(node)
            else:  # Python 3.8 or earlier
                code = ast.dump(node)
            raise CompileError(e, code, node.lineno, node.col_offset,
                               node.end_lineno, node.end_col_offset)

    def getFunctions(self):
        return self.functions

    def visit_FunctionDef(self, node):
        self.functions[node.name] = node


class CompileVisitor(ast.NodeVisitor):

    def __init__(self, m: Module, name: str, clk, rst, fsm: FSM,
                 functions: dict[str, ast.FunctionDef],
                 intrinsic_functions: dict[str, FunctionType],
                 intrinsic_methods: dict[str, MethodType],
                 start_frame: FrameType,
                 datawidth=32, point=16):

        self.m = m
        self.name: str = name
        self.clk = clk
        self.rst = rst
        self.main_fsm = fsm
        self.ram_fsms: dict[str, FSM] = dict()
        self.fsm = self.main_fsm
        self.prefetch_count = 0

        self.intrinsic_functions = intrinsic_functions
        self.intrinsic_methods = intrinsic_methods

        self.start_frame = start_frame
        self.datawidth = datawidth
        self.point = point

        self.main_scope = ScopeFrameList()
        for func in functions.values():
            self.main_scope.addFunction(func)

        self.scope = self.main_scope

        frame_scope: dict[str, Any] = dict()
        frame_scope.update(self.start_frame.f_globals)
        frame_scope.update(self.start_frame.f_locals)
        self.rams: list[str] = []
        for name, value in frame_scope.items():
            if isinstance(value, (PIPO, Inchworm)):
                self.rams.append(name)

    def get_ram_fsm(self, name: str) -> FSM:
        if name in self.ram_fsms:
            return self.ram_fsms[name]
        if name not in self.rams:
            raise KeyError
        fsm = FSM(self.m, '_'.join([self.name, name, 'fsm']), self.clk, self.rst)
        self.ram_fsms[name] = fsm
        # 0 is reserved for idle state
        fsm.state_count = 1
        return fsm

    def fork_join(self, node: ast.AST, visit: Callable[[ast.AST], None]) -> None:
        print(ast.unparse(node))
        print()

        # fork
        states: list[vtypes.Reg] = []
        for ram in self.rams:
            dma_relevant_code = extract_dma_relevant(node, ram)
            if dma_relevant_code is None:
                continue

            print(ast.unparse(dma_relevant_code))
            print()

            ram_scope = copy.deepcopy(self.main_scope)
            ram_fsm = self.get_ram_fsm(ram)

            states.append(ram_fsm.state)

            # 0 represents idle
            ram_fsm.goto_from(0, ram_fsm.current, self.main_fsm.here)
            self.main_fsm.goto_next()

            self.scope = ram_scope
            self.fsm = ram_fsm
            visit(dma_relevant_code)
            self.scope = self.main_scope
            self.fsm = self.main_fsm

            # 0 represents idle
            ram_fsm.goto_from(ram_fsm.current, 0)
            ram_fsm.inc()

        dma_irrelevant_code = extract_dma_irrelevant(node, self.rams)
        if dma_irrelevant_code is not None:
            print(ast.unparse(dma_irrelevant_code))
            print()
            visit(dma_irrelevant_code)

        # join
        if states:
            # 0 represents idle
            self.main_fsm.goto_next(vtypes.Ands(*[s == 0 for s in states]))

    # -------------------------------------------------------------------------
    def visit(self, node):
        try:
            r = super().visit(node)
            return r

        except CompileError:
            raise

        except Exception as e:
            # ast.unparse was added in Python 3.9
            if hasattr(ast, 'unparse'):  # Python 3.9 or later
                code = ast.unparse(node)
            else:  # Python 3.8 or earlier
                code = ast.dump(node)
            raise CompileError(e, code, node.lineno, node.col_offset,
                               node.end_lineno, node.end_col_offset)

    # -------------------------------------------------------------------------
    def visit_Import(self, node):
        raise TypeError("{} is not supported.".format(type(node)))

    def visit_ImportFrom(self, node):
        raise TypeError("{} is not supported.".format(type(node)))

    def visit_ClassDef(self, node):
        raise TypeError("{} is not supported.".format(type(node)))

    # -------------------------------------------------------------------------
    def visit_FunctionDef(self, node):
        raise NotImplementedError('closure is not supported.')

    def visit_Assign(self, node):
        if self.skip():
            return

        right = self.visit(node.value)
        lefts = [self.visit(target) for target in node.targets]

        for left in lefts:
            self._assign(left, right)

        self.setFsm()
        self.incFsmCount()

    def _variable_type(self, right):
        if isinstance(right, fxd._FixedBase):
            ret = {'type': 'fixed',
                   'width': max(right.get_width(), self.datawidth),
                   'point': right.point,
                   'signed': right.signed}
            return ret
        if isinstance(right, numerical_types):
            return None
        raise TypeError(f'unsupported type {type(right)}')

    def _assign(self, left, right):
        dsts = left if isinstance(left, (tuple, list)) else (left,)

        if not isinstance(right, (tuple, list)):
            if len(dsts) > 1:
                raise ValueError(
                    "too many values to unpack (expected %d)" % 1)
            _type = self._variable_type(right)
            var = self.getVariable(dsts[0], store=True, _type=_type)
            self.setAssignBind(var, right)

        elif len(dsts) == 1:
            _type = self._variable_type(right)
            var = self.getVariable(dsts[0], store=True, _type=_type)
            self.setAssignBind(var, right)

        elif len(dsts) < len(right):
            raise ValueError(
                "too many values to unpack (expected %d)" % len(right))

        elif len(dsts) > len(right):
            raise ValueError("not enough values to unpack (expected %d, got %d)" %
                             (len(dsts), len(right)))

        else:
            for d, r in zip(dsts, right):
                self._assign(d, r)

    def visit_AugAssign(self, node):
        if self.skip():
            return
        right = self.visit(node.value)
        _type = self._variable_type(right)
        left_name = self.visit(node.target)
        left = self.getVariable(left_name, store=True, _type=_type)

        if (not isinstance(left, fxd._FixedBase) and
                isinstance(right, fxd._FixedBase)):
            raise TypeError("type mismatch of operator arguments: '%s' and '%s'" %
                            (str(type(left)), str(type(right))))

        try:
            method = getMethodName(node.op)
            rslt = applyMethod(left, method, right)

        except NotImplementedError:
            op = getVeriloggenOp(node.op)
            if op is None:
                raise TypeError("unsupported BinOp: %s" % str(node.op))
            rslt = op(left, right)

        rslt = optimize(rslt)

        self.setBind(left, rslt)
        self.setFsm()
        self.incFsmCount()

    def visit_IfExp(self, node):
        test = self.visit(node.test)  # if condition
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        rslt = vtypes.Cond(test, body, orelse)
        return rslt

    def visit_If(self, node):
        if self.skip():
            return
        test = self.visit(node.test)  # if condition

        cur_count = self.getFsmCount()
        self.incFsmCount()
        true_count = self.getFsmCount()

        self.pushScope()

        for b in node.body:  # true statement
            self.visit(b)

        self.popScope()

        mid_count = self.getFsmCount()

        if len(node.orelse) == 0:
            self.setFsm(cur_count, true_count, test, mid_count)
            return

        self.incFsmCount()
        false_count = self.getFsmCount()

        self.pushScope()

        for b in node.orelse:  # false statement
            self.visit(b)

        self.popScope()

        end_count = self.getFsmCount()
        self.setFsm(cur_count, true_count, test, false_count)
        self.setFsm(mid_count, end_count)

    def visit_While(self, node):
        if self.skip():
            return

        if len(node.orelse) > 0:
            raise NotImplementedError('while-else statement is not supported.')

        # loop condition
        test = self.visit(node.test)

        begin_count = self.getFsmCount()
        self.incFsmCount()
        body_begin_count = self.getFsmCount()

        self.pushScope()

        for b in node.body:
            self.visit(b)

        self.popScope()

        body_end_count = self.getFsmCount()
        self.incFsmCount()
        loop_exit_count = self.getFsmCount()

        self.setFsm(begin_count, body_begin_count, test, loop_exit_count)
        self.setFsm(body_end_count, begin_count)

        unresolved_break = self.getUnresolvedBreak()
        for b in unresolved_break:
            self.setFsm(b, loop_exit_count)

        unresolved_continue = self.getUnresolvedContinue()
        for c in unresolved_continue:
            self.setFsm(c, begin_count)

        self.clearBreak()
        self.clearContinue()

    def visit_For(self, node: ast.For):
        if self.skip():
            return

        if not isinstance(node.target, ast.Name):
            raise NotImplementedError('unpacking in for statement is not supported.')

        if node.orelse:
            raise NotImplementedError('for-else statement is not supported.')

        if (isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == 'range'):
            return self._for_range(node)

        if isinstance(node.iter, (ast.Name, ast.Tuple, ast.List)):
            return self._for_list(node)

        raise TypeError('unsupported for-statement style')

    def _for_range(self, node: ast.For):
        if len(node.iter.args) == 0:
            raise ValueError('not enough arguments')

        if node.iter.keywords:
            raise TypeError("range() does not take keyword arguments")

        begin_node = (vtypes.Int(0)
                      if len(node.iter.args) == 1
                      else self.visit(node.iter.args[0]))

        end_node = (self.visit(node.iter.args[0])
                    if len(node.iter.args) == 1
                    else self.visit(node.iter.args[1]))

        step_node = (vtypes.Int(1)
                     if len(node.iter.args) < 3
                     else self.visit(node.iter.args[2]))

        iter_name = self.visit(node.target)
        iter_node = self.getVariable(iter_name, store=True)
        cond_node = vtypes.LessThan(iter_node, end_node)
        update_node = vtypes.Plus(iter_node, step_node)

        node_body = node.body

        return self._for_range_fsm(begin_node, end_node, step_node,
                                   iter_node, cond_node, update_node, node_body)

    def _for_range_fsm(self, begin_node, end_node, step_node,
                       iter_node, cond_node, update_node, body: list[ast.stmt],
                       target_update: tuple[Any, Any] | None = None):

        self.pushScope()

        # initialize
        self.setBind(iter_node, begin_node)
        self.setFsm()
        self.incFsmCount()

        # condition check
        check_count = self.getFsmCount()
        self.incFsmCount()
        body_begin_count = self.getFsmCount()

        # used for list/tuple access
        if target_update is not None:
            left = target_update[0]
            right = target_update[1]
            self.setBind(left, right)
            self.setFsm()
            self.incFsmCount()

        for b in body:
            self.visit(b)

        self.popScope()

        body_end_count = self.getFsmCount()

        # update
        self.setBind(iter_node, update_node)
        self.incFsmCount()
        loop_exit_count = self.getFsmCount()

        self.setFsm(body_end_count, check_count)
        self.setFsm(check_count, body_begin_count,
                    cond_node, loop_exit_count)

        unresolved_break = self.getUnresolvedBreak()
        for b in unresolved_break:
            self.setFsm(b, loop_exit_count)

        unresolved_continue = self.getUnresolvedContinue()
        for c in unresolved_continue:
            self.setFsm(c, body_end_count)

        self.clearBreak()
        self.clearContinue()

    def _for_list(self, node: ast.For):
        target_name = self.visit(node.target)
        target = self.getVariable(target_name, store=True)
        iterobj = self.visit(node.iter)

        begin_node = vtypes.Int(0)
        end_node = vtypes.Int(len(iterobj))
        step_node = vtypes.Int(1)

        iter_node = self.getTmpVariable()
        cond_node = vtypes.LessThan(iter_node, end_node)
        update_node = vtypes.Plus(iter_node, step_node)

        node_body = node.body

        patterns = []
        for i, obj in enumerate(iterobj):
            if not isinstance(obj, numerical_types):
                raise TypeError("unsupported type for for-statement")
            patterns.append((iter_node == i, obj))
        patterns.append((None, vtypes.IntX()))

        target_update = (target, vtypes.PatternMux(*patterns))

        return self._for_range_fsm(begin_node, end_node, step_node,
                                   iter_node, cond_node, update_node, node_body,
                                   target_update)

    # --------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        if self.skip():
            return

        if isinstance(node.func, ast.Name):
            return self._call_Name(node)

        if isinstance(node.func, ast.Attribute):
            return self._call_Attribute(node)

        raise NotImplementedError('%s' % str(ast.dump(node)))

    def _call_Name(self, node: ast.Call):
        name: str = node.func.id

        # system task
        if name == 'print':  # display
            return self._call_Name_print(node)
        if name == 'int':
            return self._call_Name_int(node)
        if name == 'len':
            return self._call_Name_len(node)

        # intrinsic function call
        if name in self.intrinsic_functions:
            return self._call_Name_intrinsic_function(node, name)

        # function call
        return self._call_Name_function(node, name)

    def _call_Name_print(self, node: ast.Call):
        if node.keywords:
            raise ValueError('keyword arguments for `print` built-in function are not supported')

        # prepare the argument values
        argvalues = []
        formatstring_list = []
        for arg in node.args:
            if (isinstance(arg, ast.BinOp) and
                    isinstance(arg.op, ast.Mod) and isinstance(arg.left, ast.Str)):
                # format string in print statement
                values, form = self._print_binop_mod(arg)

                for value in values:
                    if isinstance(value, fxd._FixedBase):
                        if value.point >= 0:
                            argvalues.append(vtypes.Div(vtypes.SystemTask('itor', value),
                                                        1.0 * (2 ** value.point)))
                        else:
                            argvalues.append(vtypes.Times(value, 2 ** -value.point))

                    else:
                        argvalues.append(value)

                formatstring_list.append(form)
                formatstring_list.append(" ")

            elif isinstance(arg, ast.Tuple):
                for e in arg.elts:
                    value = self.visit(e)
                    if isinstance(value, vtypes.Str):
                        formatstring_list.append(value.value)
                        formatstring_list.append(" ")
                    elif isinstance(value, fxd._FixedBase):
                        if value.point >= 0:
                            argvalues.append(vtypes.Div(vtypes.SystemTask('itor', value),
                                                        1.0 * (2 ** value.point)))
                        else:
                            argvalues.append(vtypes.Times(value, 2 ** value.point))

                        formatstring_list.append("%f")
                        formatstring_list.append(" ")
                    else:
                        argvalues.append(value)
                        formatstring_list.append("%d")
                        formatstring_list.append(" ")

            else:
                value = self.visit(arg)
                if isinstance(value, vtypes.Str):
                    formatstring_list.append(value.value)
                    formatstring_list.append(" ")
                elif isinstance(value, fxd._FixedBase):
                    if value.point >= 0:
                        argvalues.append(vtypes.Div(vtypes.SystemTask('itor', value),
                                                    1.0 * (2 ** value.point)))
                    else:
                        argvalues.append(vtypes.Times(value, 2 ** value.point))

                    formatstring_list.append("%f")
                    formatstring_list.append(" ")
                else:
                    argvalues.append(value)
                    formatstring_list.append("%d")
                    formatstring_list.append(" ")

        formatstring_list = formatstring_list[:-1]

        args = []
        args.append(vtypes.Str(''.join(formatstring_list)))
        args.extend(argvalues)

        left = None
        right = vtypes.SystemTask('display', *args)
        self.setBind(left, right)

        self.setFsm()
        self.incFsmCount()

        return right

    def _print_binop_mod(self, arg):
        values = []
        if isinstance(arg.right, ast.Tuple) or isinstance(arg.right, ast.List):
            for e in arg.right.elts:
                values.append(self.visit(e))
        else:
            values.append(self.visit(arg.right))
        form = arg.left.s
        return values, form

    def _call_Name_int(self, node: ast.Call):
        if len(node.args) > 1:
            raise TypeError(
                'takes %d positional arguments but %d were given' % (1, len(node.args)))
        argvalues = []
        for arg in node.args:
            argvalues.append(self.visit(arg))
        return argvalues[0]

    def _call_Name_len(self, node: ast.Call):
        if len(node.args) > 1:
            raise TypeError(
                'takes %d positional arguments but %d were given' % (1, len(node.args)))
        value = self.visit(node.args[0])
        if not isinstance(value, numerical_types):
            return len(value)

        ln = getattr(value, '_len', None)
        if ln is not None:
            return ln

        return vtypes.get_width(value)

    def _call_Name_function(self, node: ast.Call, name: str):
        tree = self.getFunction(name)

        # prepare the argument values
        args = []
        keywords = []
        for arg in node.args:
            args.append(self.visit(arg))
        for key in node.keywords:
            keywords.append(self.visit(key.value))

        # stack a new scope frame
        self.pushScope(ftype='call')

#        def get_resolved_args(tree, name, *args, **kwargs):
#            tree = ast.Module([tree])
#
#            code = compile(tree, 'tmp.py', 'exec')
#            exec(code)
#            targ = locals()[name]
#            resolved_args = inspect.getcallargs(targ, *args, **kwargs)
#            return resolved_args
#
#        # kwargs
#        kwargs = OrderedDict()
#        for pos, key in enumerate(node.keywords):
#            kwargs[key.arg] = keywords[pos]
#
#        resolved_args = get_resolved_args(tree, name, *args, **kwargs)
#        for arg, value in sorted(resolved_args.items(), key=lambda x: x[0]):
#            self.setArgBind(arg, value)

        args_name_list = [arg.id if isinstance(arg, ast.Name) else arg.arg
                          for arg in tree.args.args]
        args_used_list = []

        # regular args
        rest_args = []
        for pos, arg in enumerate(args):
            if pos < len(tree.args.args):
                baseobj = tree.args.args[pos]
                argname = (baseobj.id
                           if isinstance(baseobj, ast.Name)  # python 2
                           else baseobj.arg)  # python 3
                self.setArgBind(argname, arg)
                args_used_list.append(argname)
            elif tree.args.vararg is not None:
                rest_args.append(arg)
            else:
                raise TypeError('takes %d positional arguments but %d were given' %
                                (len(tree.args.args), len(args)))

        # variable length args
        if rest_args:
            baseobj = tree.args.vararg
            argname = (baseobj.id
                       if isinstance(baseobj, ast.Name)  # python 2
                       else baseobj.arg)  # python 3
            self.setVarargBind(argname, rest_args)

        # kwargs
        for pos, key in enumerate(node.keywords):
            if key.arg in args_used_list:
                raise TypeError(
                    "got multiple values for argument '%s'" % key.arg)
            if key.arg in args_name_list:
                self.setArgBind(key.arg, keywords[pos])
                args_used_list.append(key.arg)
            else:
                raise TypeError('keyword-only argument is not supported')

        # default values of kwargs
        kwargs_size = len(tree.args.defaults)
        if kwargs_size > 0:
            for arg, val in zip(tree.args.args[-kwargs_size:], tree.args.defaults):
                argname = (arg.id if isinstance(arg, ast.Name)  # python 2
                           else arg.arg)  # python 3
                if argname not in args_used_list:
                    right = self.visit(val)
                    self.setArgBind(argname, right)

        self.setFsm()
        self.incFsmCount()

        # visit the function definition
        ret = self._visit_next_function(tree)

        # fsm jump by return statement
        end_count = self.getFsmCount()
        unresolved_return = self.getUnresolvedReturn()
        for ret_count, value in unresolved_return:
            self.setFsm(ret_count, end_count)

        # clean-up jump conditions
        self.clearBreak()
        self.clearContinue()
        self.clearReturn()
        self.clearReturnVariable()

        # return to the previous scope frame
        self.popScope()

        return ret

    def _visit_next_function(self, node: ast.FunctionDef):
        for ram in self.rams:
            if extract_dma_relevant(node, ram) is not None:
                self.fork_join(node, self.generic_visit)
                return vtypes.Int(0)

        self.generic_visit(node)
        retvar = self.getReturnVariable()
        if retvar is not None:
            return retvar
        return vtypes.Int(0)

    def _call_Name_intrinsic_function(self, node: ast.Call, name: str):
        args = []
        args.append(self.fsm)
        for arg in node.args:
            args.append(self.visit(arg))

        kwargs = OrderedDict()
        for key in node.keywords:
            kwargs[key.arg] = self.visit(key.value)

        func = self.intrinsic_functions[name]

        return func(*args, **kwargs)

    def _call_Attribute(self, node: ast.Call):
        value = self.visit(node.func.value)
        method = getattr(value, node.func.attr)

        if not inspect.ismethod(method) and not inspect.isfunction(method):
            raise TypeError("'%s' object is not callable" % str(type(method)))

        # prepare the argument values
        args = []
        kwargs = OrderedDict()
        for arg in node.args:
            args.append(self.visit(arg))
        for key in node.keywords:
            kwargs[key.arg] = self.visit(key.value)

        # check intrinsic method
        name = str(method)
        if self._is_intrinsic_method(value, method) or name in self.intrinsic_methods:
            args.insert(0, self.fsm)

            # pass the current local scope
            from .thread import Thread
            from .pool import ThreadPool
            if isinstance(value, Thread):
                value.start_frame = self.start_frame
            if isinstance(value, ThreadPool):
                for thread in value.threads:
                    thread.start_frame = self.start_frame

            return method(*args, **kwargs)

        # stack a new scope frame
        self.pushScope(ftype='call')

        resolved_args = inspect.getcallargs(method, *args, **kwargs)
        for arg, value in sorted(resolved_args.items(), key=lambda x: x[0]):
            self.setArgBind(arg, value)

        self.setFsm()
        self.incFsmCount()

        text = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(text).body[0]

        # visit the function definition
        ret = self._visit_next_function(tree)

        # fsm jump by return statement
        end_count = self.getFsmCount()
        unresolved_return = self.getUnresolvedReturn()
        for ret_count, value in unresolved_return:
            self.setFsm(ret_count, end_count)

        # clean-up jump conditions
        self.clearBreak()
        self.clearContinue()
        self.clearReturn()
        self.clearReturnVariable()

        # return to the previous scope frame
        self.popScope()

        return ret

    def _is_intrinsic_method(self, value, method):
        intrinsics = getattr(value, '__intrinsics__', ())
        return (method.__name__ in intrinsics)

    # ------------------------------------------------------------------
    def visit_Nonlocal(self, node):
        raise NotImplementedError('nonlocal is not supported.')

    def visit_Global(self, node):
        raise NotImplementedError('global is not supported.')

    def visit_Pass(self, node):
        pass

    def visit_Break(self, node):
        self.addBreak()
        self.incFsmCount()

    def visit_Continue(self, node):
        self.addContinue()
        self.incFsmCount()

    def visit_Return(self, node):
        if node.value is None:
            self.addReturn(None)
            self.incFsmCount()
            return None

        retvar = self.getReturnVariable()
        if retvar is not None:
            left = retvar
            right = self.visit(node.value)
            self.setBind(left, right)
            self.addReturn(right)
            self.incFsmCount()
            return left

        tmp = self.getTmpVariable()
        self.setReturnVariable(tmp)
        left = tmp
        right = self.visit(node.value)
        self.setBind(left, right)
        self.addReturn(right)
        self.incFsmCount()
        return left

    def visit_Constant(self, node):
        if node.value is None:
            return 0
        if isinstance(node.value, int):
            return node.value
        if isinstance(node.value, float):
            v = FixedConst(None, node.value, self.point)
            v.orig_value = node.value
            return v
        if isinstance(node.value, str):
            return vtypes.Str(node.value)
        raise TypeError("%s in Const.value is not supported." %
                        str(node.value))

    def visit_Num(self, node):
        # deprecated, merged to visit_Constant
        if isinstance(node.n, int):
            # return vtypes.Int(node.n)
            return node.n
        if isinstance(node.n, float):
            v = FixedConst(None, node.n, self.point)
            v.orig_value = node.n
            return v
        return vtypes._Constant(node.n)

    def visit_Str(self, node):
        # deprecated, merged to visit_Constant
        return vtypes.Str(node.s)

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)

        try:
            method = getMethodName(node.op)
            rslt = applyMethod(value, method)
        except NotImplementedError:
            op = getVeriloggenOp(node.op)
            rslt = op(value)

        return optimize(rslt)

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]

        try:
            method = getMethodName(node.op)
            rslt = values[0]
            for v in values[1:]:
                rslt = applyMethod(rslt, method, v)

        except NotImplementedError:
            op = getVeriloggenOp(node.op)
            rslt = values[0]
            for v in values[1:]:
                rslt = op(rslt, v)

        return optimize(rslt)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(left, vtypes.Str) or isinstance(right, vtypes.Str):
            if not isinstance(node.op, ast.Add):
                raise TypeError("Cannot generate a corresponding node")
            return self._string_operation_plus(left, right)

        if (not isinstance(left, fxd._FixedBase) and
                isinstance(right, fxd._FixedBase)):
            raise TypeError("type mismatch of operator arguments: '%s' and '%s'" %
                            (str(type(left)), str(type(right))))

        try:
            method = getMethodName(node.op)
            rslt = applyMethod(left, method, right)

        except NotImplementedError:
            op = getVeriloggenOp(node.op)
            if op is None:
                raise TypeError("unsupported BinOp: %s" % str(node.op))
            rslt = op(left, right)

        return optimize(rslt)

    def _string_operation_plus(self, left, right):
        if not isinstance(left, vtypes.Str) or not isinstance(right, vtypes.Str):
            raise TypeError("'+' operation requires two string arguments")
        return vtypes.Str(left.value + right.value)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        methods = [getMethodName(op) for op in node.ops]
        ops = [getVeriloggenOp(op) for op in node.ops]
        comparators = [self.visit(comp) for comp in node.comparators]

        rslts = []
        for i, (method, op) in enumerate(zip(methods, ops)):
            if i == 0:
                if (not isinstance(left, fxd._FixedBase) and
                        isinstance(comparators[i], fxd._FixedBase)):
                    raise TypeError("type mismatch of operator arguments: '%s' and '%s'" %
                                    (str(type(left)), str(type(comparators[i]))))

                try:
                    rslts.append(applyMethod(left, method, comparators[i]))
                except NotImplementedError:
                    rslts.append(op(left, comparators[i]))
            else:
                if (not isinstance(comparators[i - 1], fxd._FixedBase) and
                        isinstance(comparators[i], fxd._FixedBase)):
                    raise TypeError("type mismatch of operator arguments: '%s' and '%s'" %
                                    (str(type(comparators[i - 1])), str(type(comparators[i]))))

                try:
                    rslts.append(
                        applyMethod(comparators[i - 1], method, comparators[i]))
                except NotImplementedError:
                    rslts.append(op(comparators[i - 1], comparators[i]))

        if len(rslts) == 1:
            return rslts[0]

        ret = None
        for r in rslts:
            if ret:
                ret = vtypes.Land(ret, r)
            else:
                ret = r
        return ret

    def visit_NameConstant(self, node):
        # deprecated, merged to visit_Constant
        if node.value is True:
            return vtypes.Int(1)
        if node.value is False:
            return vtypes.Int(0)
        if node.value is None:
            return vtypes.Int(0)
        raise TypeError("%s in NameConst.value is not supported." %
                        str(node.value))

    def visit_Name(self, node):
        store = isinstance(node.ctx, ast.Store)
        if store:
            return node.id

        name = self.getVariable(node.id)
        return name

    def visit_Print(self, node):
        # for Python 2.x
        # prepare the argument values
        argvalues = []
        formatstring_list = []
        for arg in node.values:
            if (isinstance(arg, ast.BinOp) and
                isinstance(arg.op, ast.Mod) and
                    isinstance(arg.left, ast.Str)):
                # format string in print statement
                values, form = self._print_binop_mod(arg)
                argvalues.extend(values)
                formatstring_list.append(form)
                formatstring_list.append(" ")
            elif isinstance(arg, ast.Tuple):
                for e in arg.elts:
                    value = self.visit(e)
                    if isinstance(value, vtypes.Str):
                        formatstring_list.append(value.value)
                        formatstring_list.append(" ")
                    else:
                        argvalues.append(value)
                        formatstring_list.append("%d")
                        formatstring_list.append(" ")
            else:
                value = self.visit(arg)
                if isinstance(value, vtypes.Str):
                    formatstring_list.append(value.value)
                    formatstring_list.append(" ")
                else:
                    argvalues.append(value)
                    formatstring_list.append("%d")
                    formatstring_list.append(" ")

        formatstring_list = formatstring_list[:-1]

        args = []
        args.append(vtypes.Str(''.join(formatstring_list)))
        args.extend(argvalues)

        left = None
        right = vtypes.SystemTask('display', *args)
        self.setBind(left, right)

        self.setFsm()
        self.incFsmCount()

        return right

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        attr = node.attr
        if isinstance(value, numerical_types) and attr == 'value':
            return value
        obj = getattr(value, attr)
        return obj

    def visit_Tuple(self, node):
        return tuple([self.visit(elt) for elt in node.elts])

    def visit_List(self, node):
        """ List is not fully implemented yet. """
        return tuple([self.visit(elt) for elt in node.elts])

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Slice):
            return self._subscript_slice(node)
        if isinstance(node.slice, ast.ExtSlice):
            return self._subscript_extslice(node)
        if isinstance(node.slice, ast.Index):
            return self._subscript_index(node)

        value = self.visit(node.value)
        index = self.visit(node.slice)
        index = vtypes.raw_value(optimize(index))
        return value[index]

    def _subscript_slice(self, node):
        value = self.visit(node.value)
        lower = (self.visit(node.slice.lower)
                 if node.slice.lower is not None else None)
        upper = (self.visit(node.slice.upper)
                 if node.slice.upper is not None else None)
        step = (self.visit(node.slice.step)
                if node.slice.step is not None else None)
        lower = vtypes.raw_value(optimize(lower))
        upper = vtypes.raw_value(optimize(upper))
        step = vtypes.raw_value(optimize(step))
        return value[lower:upper:step]

    def _subscript_extslice(self, node):
        value = self.visit(node.value)
        for dim in node.slice.dims:
            lower = (self.visit(dim.lower)
                     if dim.lower is not None else None)
            upper = (self.visit(dim.upper)
                     if dim.upper is not None else None)
            step = (self.visit(dim.step)
                    if dim.step is not None else None)
            lower = vtypes.raw_value(optimize(lower))
            upper = vtypes.raw_value(optimize(upper))
            step = vtypes.raw_value(optimize(step))
            value = value[lower:upper:step]
        return value

    def _subscript_index(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice.value)
        index = vtypes.raw_value(optimize(index))
        return value[index]

    # -------------------------------------------------------------------------
    def skip(self):
        val = self.hasBreak() or self.hasContinue() or self.hasReturn()
        return val

    def makeVariable(self, name, width=None, _type=None):
        if _type is None:
            return self.makeVariableReg(name, width=width)

        if _type['type'] == 'fixed':
            if width is not None:
                raise ValueError('conflicting specification of width')
            width = _type['width']
            point = _type['point']
            signed = _type['signed']
            return self.makeVariableFixed(name, width, point, signed)

        raise TypeError("not supported variable type")

    def makeVariableReg(self, name, width=None, initval=0):
        signame = _tmp_name('_'.join(['', self.name, name]))
        if width is None:
            width = self.datawidth
        return self.m.Reg(signame, width, initval=initval, signed=True)

    def makeVariableFixed(self, name, width=None, point=0, signed=True):
        signame = _tmp_name('_'.join(['', self.name, name]))
        if width is None:
            width = self.datawidth
        return fxd.FixedReg(self.m, signame, width=width, point=point, signed=signed)

    def getVariable(self, name, width=None, store=False, _type=None):
        if isinstance(name, vtypes._Numeric):
            return name

        var = self.scope.searchVariable(name)
        if var is None:
            if not store:
                local_objects = self.start_frame.f_locals
                if name in local_objects:
                    return local_objects[name]
                glb = self.getGlobalObject(name)
                if glb is not None:
                    return glb
                raise NameError("name '%s' is not defined" % name)
            var = self.makeVariable(name, width=width, _type=_type)
            self.scope.addVariable(name, var)
            var = self.scope.searchVariable(name)
        return var

    def getTmpVariable(self, width=None, _type=None):
        name = _tmp_name('tmp')
        var = self.getVariable(name, width=width, store=True, _type=_type)
        return var

    def getGlobalObject(self, name):
        global_objects = self.start_frame.f_globals
        if name in global_objects:
            return global_objects[name]
        return None

    def getFunction(self, name: str) -> ast.FunctionDef:
        func = self.scope.searchFunction(name)
        if func is not None:
            return func

        # implicit function definitions
        local_objects = self.start_frame.f_locals
        if name in local_objects:
            func = local_objects[name]
            if inspect.isfunction(func):
                text = textwrap.dedent(inspect.getsource(func))
                tree = ast.parse(text).body[0]
                return tree

        raise NameError("function '%s' is not defined" % name)

    def setArgBind(self, name, value):
        if not isinstance(value, numerical_types):
            self.scope.addVariable(name, value)
            return

        right = optimize(value)
        left = self.getVariable(name, store=True)

        self.setBind(left, right)

    def setVarargBind(self, name, values):
        lefts = []
        for value in values:
            if isinstance(value, numerical_types):
                right = optimize(value)
                left = self.getTmpVariable()

                self.setBind(left, right)
                lefts.append(left)
            else:
                lefts.append(value)
        self.scope.addVariable(name, lefts)

    def setAssignBind(self, dst, value):
        if not isinstance(value, numerical_types):
            raise TypeError("dynamic object substitution is not supported")

        self.setBind(dst, value)

    def setBind(self, var, value, cond=None):
        if var is None:
            cond = None

        if not isinstance(var, fxd._FixedVariable) and isinstance(value, fxd._FixedBase):
            raise ValueError("type mismatch of destination and source: '%s' and '%s'" %
                             (str(type(var)), str(type(value))))

        if isinstance(var, fxd._FixedVariable) and isinstance(value, fxd._FixedBase):
            if var.point != value.point:
                raise ValueError("type mismatch of fixed point: %d != %d" %
                                 (var.point, value.point))

        value = optimize(value)
        cond = optimize(cond) if cond is not None else None
        subst = (vtypes.SingleStatement(value) if var is None else
                 var.write(value))

        if var is not None:
            if hasattr(var, '_fsm') and id(var._fsm) != id(self.fsm):
                raise ValueError(
                    "variable '%s' has multiple drivers" % str(var))

            if not hasattr(var, '_fsm'):
                var._fsm = self.fsm

        self.fsm._add_statement([subst], cond=cond)

    # -------------------------------------------------------------------------
    def setFsm(self, src=None, dst=None, cond=None, else_dst=None):
        if (src is None and dst is None and cond is None and else_dst is None and
                hasattr(self.fsm, 'parallel') and self.fsm.parallel):
            return
        if src is None:
            src = self.fsm.current
        if dst is None:
            dst = src + 1
        self.fsm.goto_from(src, dst, cond, else_dst)

    def incFsmCount(self):
        if hasattr(self.fsm, 'parallel') and self.fsm.parallel:
            return
        self.fsm.inc()

    def getFsmCount(self):
        return self.fsm.current

    # -------------------------------------------------------------------------
    def getCurrentScope(self):
        return self.scope.getCurrent()

    def pushScope(self, name=None, ftype=None):
        self.scope.pushScopeFrame(name, ftype)

    def popScope(self):
        self.scope.popScopeFrame()

    # -------------------------------------------------------------------------
    def addBreak(self):
        count = self.getFsmCount()
        self.scope.addBreak(count)

    def addContinue(self):
        count = self.getFsmCount()
        self.scope.addContinue(count)

    def addReturn(self, value):
        count = self.getFsmCount()
        self.scope.addReturn(count, value)

    def hasBreak(self):
        return self.scope.hasBreak()

    def hasContinue(self):
        return self.scope.hasContinue()

    def hasReturn(self):
        return self.scope.hasReturn()

    def getUnresolvedBreak(self):
        return self.scope.getUnresolvedBreak()

    def getUnresolvedContinue(self):
        return self.scope.getUnresolvedContinue()

    def getUnresolvedReturn(self):
        return self.scope.getUnresolvedReturn()

    def setReturnVariable(self, var):
        self.scope.setReturnVariable(var)

    def getReturnVariable(self):
        return self.scope.getReturnVariable()

    def clearBreak(self):
        self.scope.clearBreak()

    def clearContinue(self):
        self.scope.clearContinue()

    def clearReturn(self):
        self.scope.clearReturn()

    def clearReturnVariable(self):
        self.scope.clearReturnVariable()
