# © Artur Czarnecki. All rights reserved.

"""NPSC-5B — bounded multi-agent fan-out architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NPSC5B_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "agent_distribution"
    / "bounded_multi_agent_fanout.py"
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
    "GraphExecutor",
    "AgentExecutor",
)

_FORBIDDEN_DIRECT_EXECUTION_CALLS = (
    "AgentExecutor",
    "NexusLoop",
    "GraphExecutor",
    "ExecutionRuntime",
    "ChildExecutionRunner",
    "DelegatedSubtaskService",
    "MultiAgentCoordinationService",
)

_FORBIDDEN_SCHEDULER_PATTERNS = (
    re.compile(r"\basyncio\.Semaphore\b"),
    re.compile(r"\basyncio\.gather\b"),
    re.compile(r"\basyncio\.create_task\b"),
    re.compile(r"\bTaskGroup\b"),
    re.compile(r"\bBoundedFanOutExecutor\b"),
    re.compile(r"\bAsyncioSemaphoreBoundedFanOutExecutor\b"),
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


def _pattern_violations(path: Path, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            violations.append(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{line}:{match.group()}",
            )
    return violations


@pytest.mark.gate
def test_npsc5b_enforces_total_fan_out_item_limit() -> None:
    source = _NPSC5B_MODULE.read_text(encoding="utf-8")
    assert "MAX_FAN_OUT_ITEMS" in source
    assert "fan-out item count exceeds platform limit" in source


@pytest.mark.gate
def test_npsc5b_validates_orchestration_outcomes() -> None:
    source = _NPSC5B_MODULE.read_text(encoding="utf-8")
    assert "_normalize_orchestration_outcomes" in source
    assert "FanOutOrchestrationContractError" in source
    assert "ORCHESTRATION_CONTRACT_VIOLATION" in source


@pytest.mark.gate
def test_npsc5b_declares_orchestration_port_boundary() -> None:
    source = _NPSC5B_MODULE.read_text(encoding="utf-8")
    assert "FanOutOrchestrationPort" in source
    assert "orchestrate_fan_out" in source
    modules = _imported_modules(_NPSC5B_MODULE)
    assert not any(module.startswith("intergrax.runtime.nexus") for module in modules)


@pytest.mark.gate
def test_npsc5b_no_legacy_supervisor_imports() -> None:
    modules = _imported_modules(_NPSC5B_MODULE)
    violations = sorted(module for module in modules if module in _FORBIDDEN_SUPERVISOR_IMPORTS)
    assert violations == [], (
        "NPSC-5B fan-out must not import legacy supervisor:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5b_no_execution_identity_minting() -> None:
    violations = _source_calls(
        _NPSC5B_MODULE,
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
        "fan-out must not mint execution identity:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5b_no_execution_ownership_operations() -> None:
    violations = _source_calls(_NPSC5B_MODULE, _FORBIDDEN_OWNERSHIP_CALLS)
    violations.extend(_instantiation_calls(_NPSC5B_MODULE, _FORBIDDEN_DIRECT_EXECUTION_CALLS))
    assert violations == [], (
        "fan-out must not own execution lifecycle:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5b_no_local_scheduler_or_nexus_imports() -> None:
    modules = _imported_modules(_NPSC5B_MODULE)
    nexus_imports = sorted(
        module for module in modules if module.startswith("intergrax.runtime.nexus")
    )
    assert nexus_imports == [], (
        "fan-out production module must not import Nexus backend:\n"
        + "\n".join(nexus_imports)
    )
    scheduler_violations = _pattern_violations(_NPSC5B_MODULE, _FORBIDDEN_SCHEDULER_PATTERNS)
    assert scheduler_violations == [], (
        "fan-out production module must not own local scheduling:\n"
        + "\n".join(scheduler_violations)
    )


@pytest.mark.gate
def test_npsc5b_no_prohibited_patterns() -> None:
    violations = _pattern_violations(_NPSC5B_MODULE, _FORBIDDEN_PROHIBITED_PATTERNS)
    assert violations == [], (
        "NPSC-5B production module contains prohibited patterns:\n"
        + "\n".join(violations)
    )
