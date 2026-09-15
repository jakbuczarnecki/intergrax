# © Artur Czarnecki. All rights reserved.

"""Contract tests for neutral terminal diagnostic integration (OBS-DIAG-PORT-1)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalDiagnosticDispatchStatus,
    TerminalExecutionDiagnosticRequest,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_DIAG_ROOT = _REPO_ROOT / "intergrax" / "contracts" / "diagnostics"


def test_terminal_execution_diagnostic_request_validates_identity_and_time() -> None:
    request = TerminalExecutionDiagnosticRequest(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        observed_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
    )
    assert request.tenant_id == "tenant-a"


def test_terminal_execution_diagnostic_request_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            observed_at=datetime(2026, 8, 26, 12, 0),
        )


def test_dispatch_status_is_transport_level_only() -> None:
    assert TerminalDiagnosticDispatchStatus.COMPLETED.value == "completed"
    assert TerminalDiagnosticDispatchStatus.FAILED_ISOLATED.value == "failed_isolated"


def test_contracts_diagnostics_package_has_no_runtime_imports() -> None:
    forbidden_prefixes = (
        "intergrax.runtime",
        "intergrax.applications",
        "intergrax.integrations",
    )
    violations: list[str] = []
    for path in _CONTRACTS_DIAG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden_prefixes:
                    if node.module.startswith(prefix):
                        violations.append(f"{rel}:{node.lineno} imports {node.module}")
    assert violations == []
