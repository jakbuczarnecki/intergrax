# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q1: closed-world ChildExecutionRunner constructor AST discovery."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

_INTERGRAX_SCAN_ROOT: Final[Path] = repo_root() / "intergrax"
_EXCLUDE_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", "tests", "docs", "examples", "benchmarks", "model_runtime_proof", "proofs", "legacy"},
)


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


def _is_child_execution_runner_constructor_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "ChildExecutionRunner"
    if isinstance(func, ast.Subscript):
        value = func.value
        if isinstance(value, ast.Name):
            return value.id == "ChildExecutionRunner"
    return False


def _enclosing_symbol(class_stack: list[str], function_stack: list[str]) -> str:
    if function_stack:
        fn = function_stack[-1]
        if class_stack:
            return f"{class_stack[-1]}.{fn}"
        return fn
    if class_stack:
        return class_stack[-1]
    return "<module>"


def _surface_key(rel_path: str, enclosing: str) -> str:
    normalized = rel_path.replace("\\", "/")
    return f"{normalized}::{enclosing}"


class _ChildRunnerConstructorVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []
        self.surfaces: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Call(self, node: ast.Call) -> None:
        if _is_child_execution_runner_constructor_call(node):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.surfaces.add(_surface_key(self._rel_path, enclosing))
        self.generic_visit(node)


def discover_child_execution_runner_constructor_surfaces() -> frozenset[str]:
    """Registry-independent production ``ChildExecutionRunner(...)`` call surfaces under ``intergrax/``."""
    discovered: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_path))
        visitor = _ChildRunnerConstructorVisitor(rel)
        visitor.visit(tree)
        discovered |= visitor.surfaces
    return frozenset(discovered)


def discover_child_execution_runner_surfaces_in_source(rel_path: str, source: str) -> frozenset[str]:
    normalized = rel_path.replace("\\", "/")
    tree = ast.parse(source, filename=normalized)
    visitor = _ChildRunnerConstructorVisitor(normalized)
    visitor.visit(tree)
    return frozenset(visitor.surfaces)


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
    "discover_child_execution_runner_constructor_surfaces",
    "discover_child_execution_runner_surfaces_in_source",
    "discover_wire_host_effective_profile_execution_roots",
    "discover_wire_host_roots_in_source",
    "profile_aware_root_forwards_child_context_inheritance",
]
