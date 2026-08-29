# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — budget ledger ownership and architecture gates."""

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
_NEXUS_LOOP_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_REGISTRY_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "budget" / "registry.py"
_LEDGER_FORBIDDEN_IN_GRAPH = frozenset(
    {
        "grant_child_budget",
        "release_child_budget",
        "consume_budget",
        "create_execution_budget_ledger",
    }
)
_FORBIDDEN_ENTRY_POINT_SYMBOLS = frozenset(
    {
        "entry_points",
        "load_execution_budget_allocation_policy",
        "resolve_execution_budget_allocation_policy",
    }
)
_FORBIDDEN_RUN_BUDGET_FIELD_ACCESS = frozenset(
    {
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "max_llm_calls",
        "max_tool_calls",
        "max_rag_invocations",
        "max_websearch_invocations",
        "max_wall_time_seconds",
        "max_planner_iterations",
        "max_replans",
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


def _collect_run_budget_field_access(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in _FORBIDDEN_RUN_BUDGET_FIELD_ACCESS:
            violations.append(f"{rel}:{node.lineno} accesses RunBudget field {node.attr}")
    return violations


def test_graph_executor_does_not_perform_ledger_arithmetic() -> None:
    violations = _collect_forbidden_calls(_GRAPH_EXECUTOR_PATH, _LEDGER_FORBIDDEN_IN_GRAPH)
    assert violations == []


def test_nexus_loop_does_not_perform_ledger_arithmetic() -> None:
    violations = _collect_forbidden_calls(_NEXUS_LOOP_PATH, _LEDGER_FORBIDDEN_IN_GRAPH)
    assert violations == []


def test_child_execution_runner_does_not_load_budget_entry_points() -> None:
    violations = _collect_forbidden_names(_CHILD_RUNNER_PATH, _FORBIDDEN_ENTRY_POINT_SYMBOLS)
    assert violations == []


def test_graph_executor_does_not_load_budget_entry_points() -> None:
    violations = _collect_forbidden_names(_GRAPH_EXECUTOR_PATH, _FORBIDDEN_ENTRY_POINT_SYMBOLS)
    assert violations == []


def test_registry_module_owns_budget_entry_point_loading() -> None:
    source = _REGISTRY_PATH.read_text(encoding="utf-8")
    assert "entry_points" in source
    assert "load_execution_budget_allocation_policy" in source


def test_child_execution_runner_does_not_inspect_run_budget_fields_directly() -> None:
    violations = _collect_run_budget_field_access(_CHILD_RUNNER_PATH)
    assert violations == []


def test_no_second_nexus_budget_ledger_module() -> None:
    nexus_budget = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "budget"
    for path in nexus_budget.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "ExecutionBudgetLedger" not in source


def test_graph_executor_does_not_construct_run_budget_for_allocation_logic() -> None:
    violations = _collect_forbidden_calls(
        _GRAPH_EXECUTOR_PATH,
        frozenset({"snapshot_root_available", "snapshot_reservation_remaining"}),
    )
    assert violations == []


def test_nexus_loop_does_not_store_mutable_ledger_on_graph_executor() -> None:
    source = _NEXUS_LOOP_PATH.read_text(encoding="utf-8")
    assert "execution_budget_ledger=" not in source
    assert "execution_budget_ledger_factory" in source


def test_child_runner_does_not_require_composition_time_ledger() -> None:
    source = _CHILD_RUNNER_PATH.read_text(encoding="utf-8")
    assert "ledger=execution_budget_ledger" not in source
