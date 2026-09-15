# © Artur Czarnecki. All rights reserved.

"""Central Diagnostics adapter for TerminalExecutionDiagnosticPort (OBS-DIAG-PORT-1)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalDiagnosticDispatchStatus,
    TerminalExecutionDiagnosticRequest,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.central_terminal_execution_diagnostic_port import (
    CentralTerminalExecutionDiagnosticPort,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from tests.unit.runtime.diagnostics.test_diagnostic_orchestrator import (
    _build_orchestrator,
    _seed_retry_violation_sequence,
)

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def test_central_port_adapts_trigger_to_neutral_request() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)
    port = CentralTerminalExecutionDiagnosticPort(trigger)

    result = port.dispatch_terminal_execution(
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
        ),
    )

    assert result is not None
    assert result.status is TerminalDiagnosticDispatchStatus.COMPLETED


def test_central_port_isolates_trigger_failure() -> None:
    failing = MagicMock(spec=TerminalExecutionDiagnosticTrigger)
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")
    port = CentralTerminalExecutionDiagnosticPort(failing)

    result = port.dispatch_terminal_execution(
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            observed_at=_OBSERVED_AT,
        ),
    )

    assert result is not None
    assert result.status is TerminalDiagnosticDispatchStatus.FAILED_ISOLATED
