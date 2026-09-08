# © Artur Czarnecki. All rights reserved.

"""NPSC-3C-F-R1 — execution identity single authority architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_NEXUS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "nexus"
_BACKGROUND_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "background_execution"
_IDENTITY_AUTHORITY_PATH = _EXECUTION_ROOT / "identity_authority.py"
_BOUNDARY_PATH = _EXECUTION_ROOT / "boundary.py"

_SCAN_ROOTS = (_EXECUTION_ROOT, _NEXUS_ROOT, _BACKGROUND_ROOT)

_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "mint_task_id",
    }
)

_MINT_OWNER_FILES = frozenset(
    {
        _IDENTITY_AUTHORITY_PATH.relative_to(_REPO_ROOT).as_posix(),
    }
)
_BIND_OWNER_FILE = _BOUNDARY_PATH.relative_to(_REPO_ROOT).as_posix()

_MINT_EXEMPT_FILES = frozenset(
    {
        "intergrax/runtime/execution/decision_finalization_conformance.py",
    }
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_scanned_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        paths.append(path)
    return paths


def _collect_forbidden_calls(
    *,
    forbidden: frozenset[str],
    allowed_files: frozenset[str],
    exempt_files: frozenset[str] = frozenset(),
) -> list[str]:
    violations: list[str] = []
    for root in _SCAN_ROOTS:
        for path in _iter_scanned_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in allowed_files or rel in exempt_files:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in forbidden:
                    violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _collect_bind_calls(*, allowed_files: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for root in _SCAN_ROOTS:
        for path in _iter_scanned_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in allowed_files:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) == "bind_active_execution_identity":
                    violations.append(f"{rel}:{node.lineno}: bind_active_execution_identity()")
    return violations


def _function_node(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _collect_mint_execution_id_calls_in_node(node: ast.AST) -> list[int]:
    lines: list[int] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _call_name(child.func) == "mint_execution_id":
            lines.append(child.lineno)
    return lines


def test_execution_runtime_mints_root_background_child_and_retry_identity() -> None:
    source = _IDENTITY_AUTHORITY_PATH.read_text(encoding="utf-8")
    assert "def mint_root_execution_identity" in source
    assert "def mint_background_transport_identity" in source
    assert "def mint_child_execution_id" in source
    assert "def mint_retry_attempt_id" in source


def test_no_identity_mint_outside_execution_runtime() -> None:
    violations = _collect_forbidden_calls(
        forbidden=_MINT_CALLS,
        allowed_files=_MINT_OWNER_FILES,
        exempt_files=_MINT_EXEMPT_FILES,
    )
    assert violations == [], (
        "execution identity mint calls must be owned by ExecutionRuntime module: "
        + ", ".join(violations)
    )


def test_no_identity_bind_outside_execution_boundary() -> None:
    violations = _collect_bind_calls(allowed_files=frozenset({_BIND_OWNER_FILE}))
    assert violations == [], (
        "bind_active_execution_identity must be owned by ExecutionBoundary: "
        + ", ".join(violations)
    )


def test_nexus_scanned_tree_has_no_identity_mint_or_bind() -> None:
    mint_violations = _collect_forbidden_calls(
        forbidden=_MINT_CALLS,
        allowed_files=frozenset(),
    )
    bind_violations = _collect_bind_calls(allowed_files=frozenset())
    nexus_prefix = "intergrax/runtime/nexus/"
    scoped_mint = [item for item in mint_violations if item.startswith(nexus_prefix)]
    scoped_bind = [item for item in bind_violations if item.startswith(nexus_prefix)]
    assert scoped_mint == [], "Nexus must not mint execution identity: " + ", ".join(scoped_mint)
    assert scoped_bind == [], "Nexus must not bind execution identity: " + ", ".join(scoped_bind)


def test_background_persistence_does_not_mint_identity() -> None:
    violations = _collect_forbidden_calls(
        forbidden=_MINT_CALLS,
        allowed_files=_MINT_OWNER_FILES,
    )
    background_prefix = "intergrax/runtime/background_execution/"
    scoped = [item for item in violations if item.startswith(background_prefix)]
    assert scoped == [], (
        "background execution persistence must not mint identity: " + ", ".join(scoped)
    )


def test_runtime_authority_module_mints_execution_id_only_in_runtime_authority_functions() -> None:
    allowed_nodes = (
        _function_node(_IDENTITY_AUTHORITY_PATH, "mint_root_execution_identity"),
        _function_node(_IDENTITY_AUTHORITY_PATH, "mint_child_execution_id"),
    )
    allowed_lines = set()
    for node in allowed_nodes:
        allowed_lines.update(_collect_mint_execution_id_calls_in_node(node))

    owner_file = _IDENTITY_AUTHORITY_PATH.relative_to(_REPO_ROOT).as_posix()
    tree = ast.parse(
        _IDENTITY_AUTHORITY_PATH.read_text(encoding="utf-8-sig"),
        filename=str(_IDENTITY_AUTHORITY_PATH),
    )
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "mint_execution_id":
            continue
        if node.lineno not in allowed_lines:
            violations.append(f"{owner_file}:{node.lineno}: mint_execution_id()")
    assert violations == [], (
        "identity_authority.py must mint ExecutionId only in runtime authority functions: "
        + ", ".join(violations)
    )
