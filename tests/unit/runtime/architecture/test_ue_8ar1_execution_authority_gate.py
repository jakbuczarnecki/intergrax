# © Artur Czarnecki. All rights reserved.

"""UE-8AR1 — GraphExecutor/NexusLoop must not mint orchestration ExecutionId."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GRAPH_EXECUTOR_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_NEXUS_LOOP_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"


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


def _collect_resolve_parent_authority_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "resolve_parent_execution_authority_for_node":
            violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def test_graph_executor_does_not_mint_execution_id() -> None:
    violations = _collect_forbidden_calls(
        _GRAPH_EXECUTOR_PATH,
        frozenset({"mint_execution_id"}),
    )
    assert violations == []


def test_graph_executor_does_not_bind_synthetic_root_identity() -> None:
    violations = _collect_forbidden_calls(
        _GRAPH_EXECUTOR_PATH,
        frozenset({"bind_active_execution_identity"}),
    )
    assert violations == []


def test_graph_executor_does_not_resolve_topology_parent_authority() -> None:
    violations = _collect_resolve_parent_authority_calls(_GRAPH_EXECUTOR_PATH)
    assert violations == []


def test_nexus_loop_does_not_mint_execution_id() -> None:
    violations = _collect_forbidden_calls(
        _NEXUS_LOOP_PATH,
        frozenset({"mint_execution_id"}),
    )
    assert violations == []
