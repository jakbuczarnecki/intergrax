# © Artur Czarnecki. All rights reserved.

"""Architecture gate — guarded process bootstrap must stay hosting-only (DG-001B R2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.runtime.diagnostics",
        "intergrax.runtime.diagnostics.diagnostic_orchestrator",
        "intergrax.contracts.execution_identity",
        "kafka",
        "celery",
        "rabbitmq",
        "message_bus",
        "local_workspace_application",
    }
)

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "TaskId",
        "RunId",
        "AttemptId",
        "ExecutionId",
    }
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "hosting"
    / "process_bootstrap.py"
)


def _collect_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(Path(__file__).resolve().parents[4]).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in _FORBIDDEN_IMPORT_MODULES
        ):
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORT_MODULES:
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    return violations


def test_process_bootstrap_import_boundaries() -> None:
    assert _collect_violations(_MODULE_PATH) == []
