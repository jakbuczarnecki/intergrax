# © Artur Czarnecki. All rights reserved.

"""Reusable AST collectors for GR-2-R3 MODEL C1 architecture gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum


class RootEngineSymbol(StrEnum):
    EXECUTION_RUNTIME = "ExecutionRuntime"
    EXECUTION_FACADE = "Execution"


@dataclass(frozen=True, slots=True)
class ArchitectureViolation:
    path: str
    line: int
    rule: str
    symbol: str

    def as_message(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class RootEngineImportBindings:
    execution_runtime_class_aliases: frozenset[str]
    execution_facade_class_aliases: frozenset[str]
    execution_runtime_module_aliases: frozenset[str]
    execution_facade_module_aliases: frozenset[str]


EXECUTION_RUNTIME_MODULE_PATHS: frozenset[str] = frozenset(
    {
        "intergrax.runtime.execution.runtime",
        "intergrax.runtime.execution",
    }
)
EXECUTION_FACADE_MODULE_PATHS: frozenset[str] = frozenset(
    {
        "intergrax.runtime.execution.facade",
        "intergrax.runtime.execution",
    }
)


def collect_root_engine_import_bindings(tree: ast.AST) -> RootEngineImportBindings:
    runtime_aliases: set[str] = set()
    facade_aliases: set[str] = set()
    runtime_module_aliases: set[str] = set()
    facade_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in EXECUTION_RUNTIME_MODULE_PATHS:
                for alias in node.names:
                    if alias.name == "ExecutionRuntime":
                        runtime_aliases.add(alias.asname or alias.name)
            if node.module in EXECUTION_FACADE_MODULE_PATHS:
                for alias in node.names:
                    if alias.name == "Execution":
                        facade_aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.execution.runtime" and alias.asname is not None:
                    runtime_module_aliases.add(alias.asname)
                if alias.name == "intergrax.runtime.execution.facade" and alias.asname is not None:
                    facade_module_aliases.add(alias.asname)

    return RootEngineImportBindings(
        execution_runtime_class_aliases=frozenset(runtime_aliases),
        execution_facade_class_aliases=frozenset(facade_aliases),
        execution_runtime_module_aliases=frozenset(runtime_module_aliases),
        execution_facade_module_aliases=frozenset(facade_module_aliases),
    )


def _attribute_root_name(expr: ast.expr) -> tuple[str, tuple[str, ...]] | None:
    attrs: list[str] = []
    current = expr
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        return current.id, tuple(reversed(attrs))
    return None


def _call_symbol_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_execution_runtime_constructor(
    call: ast.Call,
    bindings: RootEngineImportBindings,
) -> bool:
    name = _call_symbol_name(call.func)
    if name == "ExecutionRuntime" or name in bindings.execution_runtime_class_aliases:
        return True
    chain = _attribute_root_name(call.func)
    if chain is None:
        return False
    root, attrs = chain
    if attrs != ("ExecutionRuntime",):
        return False
    return root in bindings.execution_runtime_module_aliases


def _is_execution_runtime_constructor_func(
    func: ast.expr,
    bindings: RootEngineImportBindings,
) -> bool:
    return _is_execution_runtime_constructor(ast.Call(func=func, args=[], keywords=[]), bindings)


def _is_execution_facade_constructor(
    call: ast.Call,
    bindings: RootEngineImportBindings,
) -> bool:
    name = _call_symbol_name(call.func)
    if name == "Execution" or name in bindings.execution_facade_class_aliases:
        return True
    chain = _attribute_root_name(call.func)
    if chain is None:
        return False
    root, attrs = chain
    if attrs != ("Execution",):
        return False
    return root in bindings.execution_facade_module_aliases


def _is_execution_facade_constructor_func(
    func: ast.expr,
    bindings: RootEngineImportBindings,
) -> bool:
    return _is_execution_facade_constructor(ast.Call(func=func, args=[], keywords=[]), bindings)


def _resolve_module_qualified_constructor(
    func: ast.expr,
    bindings: RootEngineImportBindings,
) -> RootEngineSymbol | None:
    chain = _attribute_root_name(func)
    if chain is None:
        return None
    root, attrs = chain
    if len(attrs) != 1:
        return None
    symbol = attrs[0]
    if symbol == "ExecutionRuntime" and root in bindings.execution_runtime_module_aliases:
        return RootEngineSymbol.EXECUTION_RUNTIME
    if symbol == "Execution" and root in bindings.execution_facade_module_aliases:
        return RootEngineSymbol.EXECUTION_FACADE
    return None


def constructor_symbol_kind(
    node: ast.expr,
    bindings: RootEngineImportBindings,
) -> RootEngineSymbol | None:
    if isinstance(node, ast.Call):
        if _is_execution_runtime_constructor(node, bindings):
            return RootEngineSymbol.EXECUTION_RUNTIME
        if _is_execution_facade_constructor(node, bindings):
            return RootEngineSymbol.EXECUTION_FACADE
        return None
    if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
        return constructor_symbol_kind(node.value, bindings)
    qualified = _resolve_module_qualified_constructor(node, bindings)
    if qualified is not None:
        return qualified
    if _is_execution_runtime_constructor_func(node, bindings):
        return RootEngineSymbol.EXECUTION_RUNTIME
    if _is_execution_facade_constructor_func(node, bindings):
        return RootEngineSymbol.EXECUTION_FACADE
    return None


class _RootExecuteProvenanceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        bindings: RootEngineImportBindings,
    ) -> None:
        self._rel_path = rel_path
        self._bindings = bindings
        self._scope_stack: list[dict[str, RootEngineSymbol]] = [{}]
        self.violations: list[ArchitectureViolation] = []

    def _scope(self) -> dict[str, RootEngineSymbol]:
        return self._scope_stack[-1]

    def _push_scope(self) -> None:
        self._scope_stack.append({})

    def _pop_scope(self) -> None:
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        kind = constructor_symbol_kind(node.value, self._bindings)
        if kind is not None and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self._scope()[node.targets[0].id] = kind
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            kind = constructor_symbol_kind(node.value, self._bindings)
            if kind is not None:
                self._scope()[node.target.id] = kind
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            self._check_execute(node.func.value, node.lineno)
        self.generic_visit(node)

    def _check_execute(self, receiver: ast.expr, lineno: int) -> None:
        if isinstance(receiver, ast.Call):
            kind = constructor_symbol_kind(receiver, self._bindings)
            if kind == RootEngineSymbol.EXECUTION_RUNTIME:
                self._record(lineno, "ROOT_RUNTIME_EXECUTE", "ExecutionRuntime.execute()")
                return
            if kind == RootEngineSymbol.EXECUTION_FACADE:
                self._record(lineno, "ROOT_FACADE_EXECUTE", "Execution.execute()")
                return
        if isinstance(receiver, ast.Name):
            kind = self._scope().get(receiver.id)
            if kind == RootEngineSymbol.EXECUTION_RUNTIME:
                self._record(lineno, "ROOT_RUNTIME_EXECUTE", "ExecutionRuntime.execute()")
            elif kind == RootEngineSymbol.EXECUTION_FACADE:
                self._record(lineno, "ROOT_FACADE_EXECUTE", "Execution.execute()")

    def _record(self, lineno: int, rule: str, symbol: str) -> None:
        self.violations.append(
            ArchitectureViolation(
                path=self._rel_path,
                line=lineno,
                rule=rule,
                symbol=symbol,
            )
        )


def collect_forbidden_root_engine_execute_calls(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    bindings = collect_root_engine_import_bindings(tree)
    visitor = _RootExecuteProvenanceVisitor(rel_path=rel_path, bindings=bindings)
    visitor.visit(tree)
    return visitor.violations


def collect_forbidden_execution_runtime_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in EXECUTION_RUNTIME_MODULE_PATHS:
            for alias in node.names:
                if alias.name == "ExecutionRuntime":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_IMPORT_EXECUTION_RUNTIME",
                            symbol=alias.asname or alias.name,
                        )
                    )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.execution.runtime":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_IMPORT_EXECUTION_RUNTIME",
                            symbol=alias.asname or alias.name,
                        )
                    )
    return violations


def collect_forbidden_execution_facade_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in EXECUTION_FACADE_MODULE_PATHS:
            for alias in node.names:
                if alias.name == "Execution":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_IMPORT_EXECUTION_FACADE",
                            symbol=alias.asname or alias.name,
                        )
                    )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.execution.facade":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_IMPORT_EXECUTION_FACADE",
                            symbol=alias.asname or alias.name,
                        )
                    )
    return violations


def collect_forbidden_root_construction_calls(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_symbol_name(node.func)
        if name in {"RootExecutionOptions", "CanonicalExecutionIntakeRequest"}:
            violations.append(
                ArchitectureViolation(
                    path=rel_path,
                    line=node.lineno,
                    rule="FORBIDDEN_ROOT_CONSTRUCTION",
                    symbol=f"{name}()",
                )
            )
    return violations


def collect_forbidden_authority_resolution_calls(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_symbol_name(node.func) != "resolve_root_parent_execution_authority":
            continue
        violations.append(
            ArchitectureViolation(
                path=rel_path,
                line=node.lineno,
                rule="FORBIDDEN_AUTHORITY_RESOLUTION",
                symbol="resolve_root_parent_execution_authority()",
            )
        )
    return violations


def collect_forbidden_legacy_execute_root_task_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "intergrax.runtime.execution.orchestration":
            continue
        for alias in node.names:
            if alias.name == "execute_root_task":
                violations.append(
                    ArchitectureViolation(
                        path=rel_path,
                        line=node.lineno,
                        rule="FORBIDDEN_LEGACY_EXECUTE_ROOT_TASK_IMPORT",
                        symbol="execute_root_task",
                    )
                )
    return violations


def collect_forbidden_unified_task_runner_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.task.unified_task_runner":
            for alias in node.names:
                if alias.name == "UnifiedTaskRunner":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT",
                            symbol="UnifiedTaskRunner",
                        )
                    )
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.task":
            for alias in node.names:
                if alias.name == "UnifiedTaskRunner":
                    violations.append(
                        ArchitectureViolation(
                            path=rel_path,
                            line=node.lineno,
                            rule="FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT",
                            symbol="UnifiedTaskRunner",
                        )
                    )
    return violations
