# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q2: import-provenance ChildExecutionRunner constructor AST discovery."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

_INTERGRAX_SCAN_ROOT: Final[Path] = repo_root() / "intergrax"
_EXCLUDE_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", "tests", "docs", "examples", "benchmarks", "model_runtime_proof", "proofs", "legacy"},
)

CANONICAL_CHILD_MODULE: Final[str] = "intergrax.runtime.execution.child"
CANONICAL_CHILD_CLASS: Final[str] = "ChildExecutionRunner"
EXECUTION_PACKAGE_MODULE: Final[str] = "intergrax.runtime.execution"


def _intergrax_path_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if _EXCLUDE_DIR_NAMES.intersection(parts):
        return True
    if "docker" in parts and "runtime-context" in parts:
        return True
    return False


def _iter_intergrax_production_py_files() -> list[Path]:
    paths: list[Path] = []
    if not _INTERGRAX_SCAN_ROOT.is_dir():
        return paths
    for py_path in _INTERGRAX_SCAN_ROOT.rglob("*.py"):
        rel = py_path.relative_to(repo_root())
        if _intergrax_path_excluded(rel):
            continue
        paths.append(py_path)
    return paths


def _surface_key(rel_path: str, enclosing: str) -> str:
    normalized = rel_path.replace("\\", "/")
    return f"{normalized}::{enclosing}"


def _enclosing_symbol(class_stack: list[str], function_stack: list[str]) -> str:
    if function_stack:
        fn = function_stack[-1]
        if class_stack:
            return f"{class_stack[-1]}.{fn}"
        return fn
    if class_stack:
        return class_stack[-1]
    return "<module>"


def _call_callee_root(node: ast.Call) -> ast.expr:
    func = node.func
    if isinstance(func, ast.Subscript):
        return func.value
    return func


@dataclass
class _ImportProvenance:
    """Module-level and scoped bindings for canonical child constructor resolution."""

    constructor_names: set[str] = field(default_factory=set)
    child_module_aliases: set[str] = field(default_factory=set)
    local_child_execution_runner_class: bool = False

    def copy(self) -> _ImportProvenance:
        return _ImportProvenance(
            constructor_names=set(self.constructor_names),
            child_module_aliases=set(self.child_module_aliases),
            local_child_execution_runner_class=self.local_child_execution_runner_class,
        )


def _apply_import_from(node: ast.ImportFrom, prov: _ImportProvenance) -> None:
    if node.module == CANONICAL_CHILD_MODULE and node.level == 0:
        for alias in node.names:
            if alias.name != CANONICAL_CHILD_CLASS:
                continue
            bound = alias.asname or alias.name
            prov.constructor_names.add(bound)
        return
    if node.module == EXECUTION_PACKAGE_MODULE and node.level == 0:
        for alias in node.names:
            if alias.name != "child":
                continue
            bound = alias.asname or alias.name
            prov.child_module_aliases.add(bound)


def _apply_import(node: ast.Import, prov: _ImportProvenance) -> None:
    for alias in node.names:
        if alias.name != CANONICAL_CHILD_MODULE:
            continue
        bound = alias.asname or alias.name.split(".")[-1]
        prov.child_module_aliases.add(bound)


def _collect_module_import_provenance(tree: ast.Module) -> _ImportProvenance:
    prov = _ImportProvenance()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            _apply_import_from(node, prov)
        elif isinstance(node, ast.Import):
            _apply_import(node, prov)
        elif isinstance(node, ast.ClassDef) and node.name == CANONICAL_CHILD_CLASS:
            prov.local_child_execution_runner_class = True
            prov.constructor_names.discard(CANONICAL_CHILD_CLASS)
    return prov


def _name_refers_to_canonical_constructor(name: str, prov: _ImportProvenance) -> bool:
    if prov.local_child_execution_runner_class and name == CANONICAL_CHILD_CLASS:
        return False
    return name in prov.constructor_names


def _attribute_is_canonical_constructor(value: ast.expr, attr: str, prov: _ImportProvenance) -> bool:
    if attr != CANONICAL_CHILD_CLASS:
        return False
    if isinstance(value, ast.Name) and value.id in prov.child_module_aliases:
        return True
    return False


def _expr_is_canonical_constructor_ref(expr: ast.expr, prov: _ImportProvenance) -> bool:
    if isinstance(expr, ast.Name):
        return _name_refers_to_canonical_constructor(expr.id, prov)
    if isinstance(expr, ast.Attribute):
        return _attribute_is_canonical_constructor(expr.value, expr.attr, prov)
    return False


def _is_canonical_constructor_call(node: ast.Call, prov: _ImportProvenance) -> bool:
    root = _call_callee_root(node)
    if isinstance(root, ast.Name):
        return _name_refers_to_canonical_constructor(root.id, prov)
    if isinstance(root, ast.Attribute):
        return _attribute_is_canonical_constructor(root.value, root.attr, prov)
    return False


def _assignment_targets_canonical_rebind(
    targets: list[ast.expr],
    value: ast.expr,
    prov: _ImportProvenance,
) -> list[str]:
    if not _expr_is_canonical_constructor_ref(value, prov):
        return []
    violations: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            violations.append(target.id)
    return violations


@dataclass
class ChildRunnerDiscoveryResult:
    surfaces: set[str] = field(default_factory=set)
    rebind_violations: list[str] = field(default_factory=list)


class _ChildRunnerDiscoveryVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, module_prov: _ImportProvenance) -> None:
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []
        self._scope_stack: list[_ImportProvenance] = [module_prov.copy()]
        self.result = ChildRunnerDiscoveryResult()

    def _current_prov(self) -> _ImportProvenance:
        return self._scope_stack[-1]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == CANONICAL_CHILD_CLASS:
            scoped = self._current_prov().copy()
            scoped.local_child_execution_runner_class = True
            scoped.constructor_names.discard(CANONICAL_CHILD_CLASS)
            self._scope_stack.append(scoped)
        else:
            self._scope_stack.append(self._current_prov().copy())
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(self._current_prov().copy())
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Assign(self, node: ast.Assign) -> None:
        prov = self._current_prov()
        for name in _assignment_targets_canonical_rebind(node.targets, node.value, prov):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: rebind {name}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            prov = self._current_prov()
            for name in _assignment_targets_canonical_rebind([node.target], node.value, prov):
                enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
                self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: rebind {name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_canonical_constructor_call(node, self._current_prov()):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.surfaces.add(_surface_key(self._rel_path, enclosing))
        self.generic_visit(node)


def analyze_child_execution_runner_discovery(rel_path: str, source: str) -> ChildRunnerDiscoveryResult:
    normalized = rel_path.replace("\\", "/")
    tree = ast.parse(source, filename=normalized)
    if not isinstance(tree, ast.Module):
        return ChildRunnerDiscoveryResult()
    module_prov = _collect_module_import_provenance(tree)
    visitor = _ChildRunnerDiscoveryVisitor(normalized, module_prov)
    visitor.visit(tree)
    return visitor.result


def discover_child_execution_runner_constructor_surfaces() -> frozenset[str]:
    """Registry-independent canonical ``ChildExecutionRunner(...)`` surfaces under ``intergrax/``."""
    discovered: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        discovered |= analyze_child_execution_runner_discovery(rel, text).surfaces
    return frozenset(discovered)


def discover_child_execution_runner_surfaces_in_source(rel_path: str, source: str) -> frozenset[str]:
    return frozenset(analyze_child_execution_runner_discovery(rel_path, source).surfaces)


def discover_canonical_constructor_rebinding_violations() -> list[str]:
    violations: list[str] = []
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        violations.extend(analyze_child_execution_runner_discovery(rel, text).rebind_violations)
    return violations


def discover_canonical_constructor_rebindings_in_source(rel_path: str, source: str) -> list[str]:
    return analyze_child_execution_runner_discovery(rel_path, source).rebind_violations


class _WireHostProfileVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self._rel_path = rel_path
        self._function_stack: list[str] = []
        self.roots: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "wire_host_effective_profile_execution":
            fn = self._function_stack[-1] if self._function_stack else "<module>"
            self.roots.add(_surface_key(self._rel_path, fn))
        self.generic_visit(node)


def discover_wire_host_effective_profile_execution_roots() -> frozenset[str]:
    """Production call sites of ``wire_host_effective_profile_execution(...)`` under ``intergrax/``."""
    discovered: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_path))
        visitor = _WireHostProfileVisitor(rel)
        visitor.visit(tree)
        discovered |= visitor.roots
    return frozenset(discovered)


def discover_wire_host_roots_in_source(rel_path: str, source: str) -> frozenset[str]:
    normalized = rel_path.replace("\\", "/")
    tree = ast.parse(source, filename=normalized)
    visitor = _WireHostProfileVisitor(normalized)
    visitor.visit(tree)
    return frozenset(visitor.roots)


def profile_aware_root_forwards_child_context_inheritance(source: str, root_surface_key: str) -> bool:
    """
    Mechanical check: enclosing function wires ``host_profile.child_context_inheritance``
    into host orchestration spec construction.
    """
    _path, enclosing = root_surface_key.split("::", 1)
    tree = ast.parse(source)
    target_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == enclosing:
            target_fn = node
            break
    if target_fn is None:
        return False
    for node in ast.walk(target_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        call_name = None
        if isinstance(func, ast.Name):
            call_name = func.id
        elif isinstance(func, ast.Attribute):
            call_name = func.attr
        if call_name != "build_host_orchestration_loop_init_spec_from_environment":
            continue
        for keyword in node.keywords:
            if keyword.arg != "child_context_inheritance":
                continue
            value = keyword.value
            if isinstance(value, ast.Attribute) and value.attr == "child_context_inheritance":
                if isinstance(value.value, ast.Name) and value.value.id == "host_profile":
                    return True
    return False


__all__ = [
    "CANONICAL_CHILD_CLASS",
    "CANONICAL_CHILD_MODULE",
    "ChildRunnerDiscoveryResult",
    "analyze_child_execution_runner_discovery",
    "discover_canonical_constructor_rebinding_violations",
    "discover_canonical_constructor_rebindings_in_source",
    "discover_child_execution_runner_constructor_surfaces",
    "discover_child_execution_runner_surfaces_in_source",
    "discover_wire_host_effective_profile_execution_roots",
    "discover_wire_host_roots_in_source",
    "profile_aware_root_forwards_child_context_inheritance",
]
