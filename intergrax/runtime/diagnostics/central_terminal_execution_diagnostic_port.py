# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Central Diagnostics adapter for :class:`TerminalExecutionDiagnosticPort`."""

from __future__ import annotations

from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalDiagnosticDispatchResult,
    TerminalDiagnosticDispatchStatus,
    TerminalExecutionDiagnosticPort,
    TerminalExecutionDiagnosticRequest,
)
from intergrax.runtime.diagnostics import terminal_execution_diagnostic_bridge as _terminal_diagnostic_bridge
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding


class CentralTerminalExecutionDiagnosticPort:
    """
    Platform-owned port implementation: neutral request → trigger → orchestrator.

    Failure isolation and subsystem failure evidence remain in the diagnostic bridge.
    """

    def __init__(
        self,
        trigger: TerminalExecutionDiagnosticTrigger,
        *,
        event_bus: RuntimeEventBus | None = None,
    ) -> None:
        self._trigger = trigger
        self._event_bus = event_bus

    def dispatch_terminal_execution(
        self,
        request: TerminalExecutionDiagnosticRequest,
    ) -> TerminalDiagnosticDispatchResult | None:
        execution_identity: ExecutionIdentityBinding | None = None
        if request.attempt_id is not None:
            execution_identity = ExecutionIdentityBinding(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                execution_id=request.execution_id,
            )
        orchestration_result = _terminal_diagnostic_bridge.invoke_terminal_execution_diagnostics(
            self._trigger,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            run_id=request.run_id,
            observed_at=request.observed_at,
            event_bus=self._event_bus,
            execution_identity=execution_identity,
        )
        if orchestration_result is None:
            return TerminalDiagnosticDispatchResult(
                status=TerminalDiagnosticDispatchStatus.FAILED_ISOLATED,
            )
        return TerminalDiagnosticDispatchResult(
            status=TerminalDiagnosticDispatchStatus.COMPLETED,
        )


def wrap_terminal_execution_diagnostic_trigger(
    trigger: TerminalExecutionDiagnosticTrigger,
    *,
    event_bus: RuntimeEventBus | None = None,
) -> TerminalExecutionDiagnosticPort:
    """Expose production trigger through the neutral terminal diagnostic port."""
    return CentralTerminalExecutionDiagnosticPort(trigger=trigger, event_bus=event_bus)


__all__ = [
    "CentralTerminalExecutionDiagnosticPort",
    "wrap_terminal_execution_diagnostic_trigger",
]
