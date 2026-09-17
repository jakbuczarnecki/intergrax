# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-time validation for ``RuntimeEventHistoryBuffer`` plugins."""

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.runtime_event_history import (
    RuntimeEventHistoryBuffer,
    RuntimeEventHistoryRetention,
)


def _probe_event(label: str) -> RuntimeEvent:
    return RuntimeEvent.model_validate(
        {
            "tenant_id": "history-retention-probe",
            "task_id": mint_task_id(),
            "run_id": mint_run_id(),
            "attempt_id": mint_attempt_id(),
            "execution_id": mint_execution_id(),
            "event_id": mint_event_id(),
            "event_type": RuntimeEventType.STEP_STARTED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "payload": {"probe": label},
        },
    )


def validate_runtime_event_history_buffer(buffer: RuntimeEventHistoryBuffer) -> None:
    """Reject custom buffers that do not declare bounded/disabled finite retention."""
    retention = buffer.retention()
    if not isinstance(retention, RuntimeEventHistoryRetention):
        raise ValueError(
            "history buffer retention must be RuntimeEventHistoryRetention"
        )
    if retention.mode == "disabled":
        for index in range(3):
            buffer.append(_probe_event(f"disabled-{index}"))
        if buffer.snapshot():
            raise ValueError("disabled history buffer retained events after append")
        buffer.clear()
        return

    assert retention.capacity is not None
    capacity = retention.capacity
    overflow = capacity + 2
    for index in range(overflow):
        buffer.append(_probe_event(f"bounded-{index}"))
    if len(buffer.snapshot()) > capacity:
        raise ValueError(
            "history buffer snapshot exceeds declared bounded retention capacity",
        )
    buffer.clear()


__all__ = ["validate_runtime_event_history_buffer"]
