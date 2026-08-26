# © Artur Czarnecki. All rights reserved.

"""Architecture gate — hosting must not bypass platform observability diagnostics bridge (HOST-DIAG-1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "TerminalExecutionDiagnosticTrigger",
        "invoke_terminal_execution_diagnostics",
        "ProblemLifecycleEngine",
        "ProblemGroupingEngine",
        "RuntimeSpineHostedApplicationEventPublisher",
    }
)

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.contracts.execution_identity",
        "intergrax.runtime.diagnostics.diagnostic_orchestrator",
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger",
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge",
        "intergrax.runtime.diagnostics.problem_lifecycle",
        "intergrax.runtime.diagnostics.problem_grouping",
    }
)

_SCAN_ROOT = "intergrax/hosting"

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "tests",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _hosting_python_files() -> list[Path]:
    root = _repo_root()
    base = root / _SCAN_ROOT
    files: list[Path] = []
    for path in base.rglob("*.py"):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def _collect_forbidden_symbol_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_IMPORT_MODULES:
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
    return violations


def test_hosting_cannot_bypass_platform_observability_diagnostics_bridge() -> None:
    violations: list[str] = []
    for path in _hosting_python_files():
        violations.extend(_collect_forbidden_symbol_references(path))
    assert violations == []
