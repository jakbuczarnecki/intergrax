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
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.central_terminal_execution_diagnostic_port import (
    CentralTerminalExecutionDiagnosticPort,
)
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding
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


def test_central_port_run_only_passes_no_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ExecutionIdentityBinding | None] = []

    def _capture_invoke(*_args: object, **kwargs: object) -> object:
        identity = kwargs.get("execution_identity")
        if identity is None or isinstance(identity, ExecutionIdentityBinding):
            captured.append(identity)
        else:
            captured.append(None)
        return None

    from intergrax.runtime.diagnostics import (
        terminal_execution_diagnostic_bridge as bridge_module,
    )

    monkeypatch.setattr(
        bridge_module,
        "invoke_terminal_execution_diagnostics",
        _capture_invoke,
    )
    port = CentralTerminalExecutionDiagnosticPort(MagicMock(spec=TerminalExecutionDiagnosticTrigger))

    port.dispatch_terminal_execution(
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            observed_at=_OBSERVED_AT,
        ),
    )

    assert captured == [None]


def test_central_port_full_pair_passes_execution_identity_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ExecutionIdentityBinding | None] = []

    def _capture_invoke(*_args: object, **kwargs: object) -> object:
        identity = kwargs.get("execution_identity")
        if isinstance(identity, ExecutionIdentityBinding):
            captured.append(identity)
        else:
            captured.append(None)
        return None

    from intergrax.runtime.diagnostics import (
        terminal_execution_diagnostic_bridge as bridge_module,
    )

    monkeypatch.setattr(
        bridge_module,
        "invoke_terminal_execution_diagnostics",
        _capture_invoke,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    port = CentralTerminalExecutionDiagnosticPort(MagicMock(spec=TerminalExecutionDiagnosticTrigger))

    port.dispatch_terminal_execution(
        TerminalExecutionDiagnosticRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=run_id,
            observed_at=_OBSERVED_AT,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )

    assert len(captured) == 1
    binding = captured[0]
    assert binding is not None
    assert binding.run_id == run_id
    assert binding.attempt_id == attempt_id
    assert binding.execution_id == execution_id


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
