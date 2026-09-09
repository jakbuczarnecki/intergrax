# © Artur Czarnecki. All rights reserved.

"""NPSC-5C/R2 — decision coordination projection architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROJECTION_PATH = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "decision_coordination_projection.py"
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.execution",
    "intergrax.runtime.nexus",
    "intergrax.decision_system",
)

_FORBIDDEN_IMPORT_MODULES = (
    "intergrax.agent_distribution.delegated_subtasks",
    "intergrax.agent_distribution.bounded_multi_agent_fanout",
    "intergrax.agent_distribution.task_scoped_agents",
)

_FORBIDDEN_SYMBOLS = (
    "MultiAgentCoordinationService",
    "DelegatedSubtaskService",
    "BoundedMultiAgentFanOutService",
    "TaskScopedAgentService",
    "TaskCapabilityResolver",
    "TaskCapabilityResolutionRequest",
    "TaskKind",
    "CoordinationIntentExecutor",
)

_FORBIDDEN_PATTERNS = (
    re.compile(r"\bAny\b"),
    re.compile(r"\bgetattr\("),
    re.compile(r"\bsetattr\("),
    re.compile(r"\bhasattr\("),
    re.compile(r"\bjson\.loads\("),
    re.compile(r"\bjson\.dumps\("),
    re.compile(r"\buuid4\("),
    re.compile(r"\brandom\("),
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


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


@pytest.mark.gate
def test_npsc5c_r2_projection_module_exists() -> None:
    assert _PROJECTION_PATH.is_file()


@pytest.mark.gate
def test_npsc5c_r2_projection_no_execution_or_nexus_imports() -> None:
    violations = [
        module
        for module in sorted(_imported_modules(_PROJECTION_PATH))
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
        )
    ]
    assert violations == [], (
        "decision coordination projection must not import execution/nexus/decision internals:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_r2_projection_no_execution_service_imports() -> None:
    modules = _imported_modules(_PROJECTION_PATH)
    violations = [
        module
        for module in sorted(modules)
        if module in _FORBIDDEN_IMPORT_MODULES
        or any(module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_MODULES)
    ]
    assert violations == [], (
        "decision coordination projection must not import execution services:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_r2_projection_no_task_resolver_symbols() -> None:
    source = _source(_PROJECTION_PATH)
    violations = [symbol for symbol in _FORBIDDEN_SYMBOLS if symbol in source]
    assert violations == [], (
        "decision coordination projection must not reference forbidden symbols:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_r2_projection_no_prohibited_patterns() -> None:
    source = _source(_PROJECTION_PATH)
    violations: list[str] = []
    for pattern in _FORBIDDEN_PATTERNS:
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            violations.append(f"{_PROJECTION_PATH.name}:{line}:{match.group()}")
    assert violations == [], (
        "decision coordination projection contains prohibited patterns:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_r2_projection_uses_public_decision_contract_only() -> None:
    modules = _imported_modules(_PROJECTION_PATH)
    decision_modules = sorted(
        module
        for module in modules
        if module.startswith("intergrax.contracts.decision_")
    )
    assert decision_modules == [
        "intergrax.contracts.decision_coordination",
        "intergrax.contracts.decision_identity",
        "intergrax.contracts.decision_record",
    ]


@pytest.mark.gate
def test_npsc5c_r2_projection_shape_mapping_fail_closed() -> None:
    source = _source(_PROJECTION_PATH)
    assert "unsupported DecisionCoordinationShape" in source
    assert "raise DecisionCoordinationProjectionError" in source
    function_match = re.search(
        r"def _project_execution_mode\((?:.|\n)*?\n(?=def |\Z)",
        source,
    )
    assert function_match is not None
    function_source = function_match.group(0)
    assert "else:" not in function_source
    assert "default:" not in function_source
