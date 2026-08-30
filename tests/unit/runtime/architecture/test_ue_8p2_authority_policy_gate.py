# © Artur Czarnecki. All rights reserved.

"""UE-8P2 — authority narrowing belongs to policy, not ChildExecutionRunner/GraphExecutor."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHILD_RUNNER_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_GRAPH_EXECUTOR_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_REGISTRY_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "authority" / "registry.py"
)
_FORBIDDEN_CALLS = frozenset(
    {
        "mint_effective_delegation_authority",
        "effective_delegation_to_parent_authority",
    }
)
_FORBIDDEN_ENTRY_POINT_SYMBOLS = frozenset(
    {
        "entry_points",
        "load_execution_authority_policy",
        "resolve_execution_authority_policy",
    }
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def _collect_forbidden_names(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            violations.append(f"{rel}:{node.lineno} references {node.attr}")
    return violations


def test_child_execution_runner_does_not_mint_delegation_authority() -> None:
    violations = _collect_forbidden_calls(_CHILD_RUNNER_PATH, _FORBIDDEN_CALLS)
    assert violations == []


def test_graph_executor_does_not_mint_delegation_authority() -> None:
    violations = _collect_forbidden_calls(_GRAPH_EXECUTOR_PATH, _FORBIDDEN_CALLS)
    assert violations == []


def test_child_execution_runner_does_not_load_authority_entry_points() -> None:
    violations = _collect_forbidden_names(_CHILD_RUNNER_PATH, _FORBIDDEN_ENTRY_POINT_SYMBOLS)
    assert violations == []


def test_graph_executor_does_not_load_authority_entry_points() -> None:
    violations = _collect_forbidden_names(_GRAPH_EXECUTOR_PATH, _FORBIDDEN_ENTRY_POINT_SYMBOLS)
    assert violations == []


def test_registry_module_owns_entry_point_loading() -> None:
    source = _REGISTRY_PATH.read_text(encoding="utf-8")
    assert "entry_points" in source
    assert "load_execution_authority_policy" in source
