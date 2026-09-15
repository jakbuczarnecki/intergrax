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
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

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
    assert request.attempt_id is None
    assert request.execution_id is None


def test_terminal_execution_diagnostic_request_accepts_full_correlation_pair() -> None:
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    request = TerminalExecutionDiagnosticRequest(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        observed_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert request.attempt_id == attempt_id
    assert request.execution_id == execution_id


@pytest.mark.parametrize(
    ("attempt_id", "execution_id"),
    [
        (mint_attempt_id(), None),
        (None, mint_execution_id()),
    ],
)
def test_terminal_execution_diagnostic_request_rejects_partial_correlation(
    attempt_id: object,
    execution_id: object,
) -> None:
    with pytest.raises(ValueError, match="attempt_id and execution_id"):
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            observed_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


@pytest.mark.parametrize(
    "tenant_id",
    ["", "   ", " tenant-a", "tenant-a "],
)
def test_terminal_execution_diagnostic_request_rejects_non_canonical_tenant(
    tenant_id: str,
) -> None:
    with pytest.raises(ValueError):
        TerminalExecutionDiagnosticRequest(
            tenant_id=tenant_id,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            observed_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
        )


def test_terminal_execution_diagnostic_request_enforces_pair_integrity_in_source() -> None:
    contract_path = (
        _REPO_ROOT
        / "intergrax"
        / "contracts"
        / "diagnostics"
        / "terminal_execution_diagnostic_port.py"
    )
    source = contract_path.read_text(encoding="utf-8")
    assert "attempt_id is None" in source
    assert "execution_id is None" in source
    assert "model_validator" in source


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
