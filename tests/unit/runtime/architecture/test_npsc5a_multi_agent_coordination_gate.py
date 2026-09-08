# © Artur Czarnecki. All rights reserved.

"""NPSC-5A — multi-agent coordination architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NPSC5A_MODULE = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "multi_agent_coordination.py"
)

_FORBIDDEN_SUPERVISOR_IMPORTS = (
    "intergrax.supervisor",
    "intergrax.supervisor.supervisor",
    "intergrax.supervisor.supervisor_to_state_graph",
)

_FORBIDDEN_OWNERSHIP_CALLS = (
    "mint_root_execution_identity",
    "mint_child_execution_identity",
    "bind_active_execution_identity",
    "ExecutionRuntime",
    "StrategyExecutionRouter",
    "NexusLoop",
    "AgentExecutor",
)

_FORBIDDEN_DIRECT_EXECUTION_CALLS = (
    "AgentExecutor",
    "NexusLoop",
    "ExecutionRuntime",
)

_FORBIDDEN_PROHIBITED_PATTERNS = (
    re.compile(r"\bAny\b"),
    re.compile(r"dict\[str,\s*Any\]"),
    re.compile(r"Dict\[str,\s*Any\]"),
    re.compile(r"Callable\[\.\.\."),
    re.compile(r"\bcast\("),
    re.compile(r"#\s*type:\s*ignore"),
    re.compile(r"\bgetattr\("),
    re.compile(r"\bsetattr\("),
    re.compile(r"\bhasattr\("),
    re.compile(r"\bimport\s+inspect\b"),
    re.compile(r"\bfrom\s+inspect\b"),
    re.compile(r"except\s+Exception\b"),
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _source_calls(path: Path, names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in names:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in names:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return hits


def _instantiation_calls(path: Path, class_names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in class_names:
            hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return hits


@pytest.mark.gate
def test_npsc5a_no_legacy_supervisor_imports() -> None:
    modules = _imported_modules(_NPSC5A_MODULE)
    violations = sorted(module for module in modules if module in _FORBIDDEN_SUPERVISOR_IMPORTS)
    assert violations == [], (
        "NPSC-5A coordination must not import legacy supervisor:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5a_no_langgraph_dependency() -> None:
    source = _NPSC5A_MODULE.read_text(encoding="utf-8")
    assert "langgraph" not in source.lower()


@pytest.mark.gate
def test_npsc5a_no_execution_identity_minting() -> None:
    violations = _source_calls(
        _NPSC5A_MODULE,
        (
            "mint_run_id",
            "mint_attempt_id",
            "mint_execution_id",
            "mint_task_id",
            "mint_root_execution_identity",
            "mint_child_execution_identity",
        ),
    )
    assert violations == [], (
        "coordination must not mint execution identity:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5a_no_execution_ownership_operations() -> None:
    violations = _source_calls(_NPSC5A_MODULE, _FORBIDDEN_OWNERSHIP_CALLS)
    violations.extend(_instantiation_calls(_NPSC5A_MODULE, _FORBIDDEN_DIRECT_EXECUTION_CALLS))
    assert violations == [], (
        "coordination must not own execution lifecycle:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5a_no_orchestration_backend_access() -> None:
    source = _NPSC5A_MODULE.read_text(encoding="utf-8")
    assert "_orchestration_backend" not in source


@pytest.mark.gate
def test_npsc5a_delegates_through_delegated_subtask_service() -> None:
    source = _NPSC5A_MODULE.read_text(encoding="utf-8")
    assert "DelegatedSubtaskService" in source
    assert "await self._delegated_subtasks.execute" in source
    assert "ChildExecutionPort" not in source
    assert "ChildExecutionRunner" not in source


@pytest.mark.gate
def test_npsc5a_no_prohibited_patterns() -> None:
    source = _NPSC5A_MODULE.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern in _FORBIDDEN_PROHIBITED_PATTERNS:
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            violations.append(
                f"{_NPSC5A_MODULE.relative_to(_REPO_ROOT).as_posix()}:{line}:{match.group()}"
            )
    assert violations == [], (
        "NPSC-5A production module contains prohibited patterns:\n"
        + "\n".join(violations)
    )
