# © Artur Czarnecki. All rights reserved.

"""Architecture gate — scenario reasoning must not own central diagnostic reads (DIAG-8C)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_RUNTIME_SYMBOLS = frozenset(
    {
        "ProblemPersistence",
        "ExecutionReconstructor",
        "ProblemGroupingEngine",
        "ProblemLifecycleEngine",
        "DiagnosticOrchestrator",
    }
)

_FORBIDDEN_RUNTIME_MODULES = frozenset(
    {
        "intergrax.runtime.diagnostics.problem_persistence",
        "intergrax.runtime.diagnostics.execution_reconstruction",
        "intergrax.runtime.diagnostics.problem_grouping",
        "intergrax.runtime.diagnostics.diagnostic_orchestrator",
    }
)

_REASONING_MODULES = frozenset(
    {
        "investigator_agent.py",
        "incident_reasoning.py",
        "domain_reasoning.py",
        "evidence_gathering.py",
    }
)

_COMPOSITION_ALLOWED_READ_SERVICE = "scenario_composition.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _scenario_root() -> Path:
    return _repo_root() / "platform_proofs" / "scenarios" / "ai_incident_investigation"


def _collect_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_RUNTIME_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_RUNTIME_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_RUNTIME_MODULES:
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_RUNTIME_MODULES:
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
    return violations


def _collect_diagnostic_read_service_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "diagnostic_read_service" in node.module:
                violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "diagnostic_read_service" in alias.name:
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
    return violations


def test_scenario_runtime_modules_do_not_import_central_diagnostic_ownership() -> None:
    violations: list[str] = []
    for path in sorted(_scenario_root().glob("*.py")):
        if path.name == _COMPOSITION_ALLOWED_READ_SERVICE:
            continue
        violations.extend(_collect_violations(path))
    assert violations == []


def test_reasoning_modules_do_not_import_diagnostic_read_service() -> None:
    violations: list[str] = []
    for name in _REASONING_MODULES:
        path = _scenario_root() / name
        violations.extend(_collect_diagnostic_read_service_imports(path))
    assert violations == []


def test_composition_boundary_may_import_diagnostic_read_service() -> None:
    path = _scenario_root() / _COMPOSITION_ALLOWED_READ_SERVICE
    source = path.read_text(encoding="utf-8")
    assert "DiagnosticReadService" in source
