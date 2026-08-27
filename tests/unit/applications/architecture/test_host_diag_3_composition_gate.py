# © Artur Czarnecki. All rights reserved.

"""Architecture gate — HOST-DIAG-3 hosted diagnostic composition boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOSTING_ROOT = _REPO_ROOT / "intergrax" / "hosting"
_HOSTED_DIAG_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "hosted_application_diagnostic_wiring.py"
)
_HOSTED_FAILURE_PROJECTION = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "hosted_application_failure_projection.py"
)

_FORBIDDEN_HOSTING_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemLifecycleEngine",
        "ProblemGroupingEngine",
        "TerminalExecutionDiagnosticTrigger",
    },
)

_FORBIDDEN_WIRING_SYMBOLS = frozenset(
    {
        "InMemoryProblemPersistence",
        "wire_problem_persistence",
        "DocumentStoreProblemPersistence",
    },
)


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        files.append(path)
    return files


def _collect_symbol_refs(path: Path, symbols: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in symbols:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in symbols:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    return violations


def test_hosting_core_has_no_direct_diagnostic_construction() -> None:
    violations: list[str] = []
    for path in _python_files(_HOSTING_ROOT):
        violations.extend(_collect_symbol_refs(path, _FORBIDDEN_HOSTING_SYMBOLS))
    assert violations == []


def test_hosted_diagnostic_wiring_uses_shared_orchestrator_not_private_persistence() -> None:
    violations = _collect_symbol_refs(_HOSTED_DIAG_WIRING, _FORBIDDEN_WIRING_SYMBOLS)
    assert violations == []


def test_hosted_failure_projection_has_no_synthetic_execution_identity() -> None:
    source = _HOSTED_FAILURE_PROJECTION.read_text(encoding="utf-8")
    assert "task_id" not in source.replace("task_id ==", "")
    assert "run_id" not in source.replace("run_id ==", "")


def test_hosted_diagnostic_wiring_rejects_application_id_as_tenant() -> None:
    source = _HOSTED_DIAG_WIRING.read_text(encoding="utf-8")
    assert "tenant_id=event.application_id" not in source
    assert "tenant_id = event.application_id" not in source
    assert "tenant_id=application_id" not in source
