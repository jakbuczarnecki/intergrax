# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-time validation for platform-owned history buffers."""

from __future__ import annotations

import hashlib

from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_event_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.runtime_event_history import (
    RuntimeEventHistoryBuffer,
    RuntimeEventHistoryRetention,
)
_PROBE_NAMESPACE = "history-retention-probe"


def _probe_id_suffix(seed: str) -> str:
    digest = hashlib.sha256(f"{_PROBE_NAMESPACE}:{seed}".encode("utf-8")).hexdigest()
    return digest[:32]


def _probe_event(label: str) -> RuntimeEvent:
    suffix = _probe_id_suffix(label)
    return RuntimeEvent.model_validate(
        {
            "tenant_id": _PROBE_NAMESPACE,
            "task_id": validate_task_id(f"task_{suffix}"),
            "run_id": validate_run_id(f"run_{suffix}"),
            "attempt_id": validate_attempt_id(f"attempt_{suffix}"),
            "execution_id": validate_execution_id(f"exec_{suffix}"),
            "event_id": validate_event_id(f"evt_{suffix}"),
            "event_type": RuntimeEventType.STEP_STARTED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "payload": {"probe": label},
        },
    )


def validate_platform_runtime_event_history_buffer(
    buffer: RuntimeEventHistoryBuffer,
) -> None:
    """Verify platform-owned retention mechanics (not external plugin honesty)."""
    from intergrax.runtime.events.runtime_event_history import (
        PlatformOwnedRuntimeEventHistoryBuffer,
    )

    if not isinstance(buffer, PlatformOwnedRuntimeEventHistoryBuffer):
        raise TypeError(
            "runtime event history buffer must be PlatformOwnedRuntimeEventHistoryBuffer",
        )
    retention = buffer.retention()
    if not isinstance(retention, RuntimeEventHistoryRetention):
        raise ValueError(
            "history buffer retention must be RuntimeEventHistoryRetention",
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
            "platform history buffer snapshot exceeds declared bounded retention capacity",
        )
    buffer.clear()


validate_runtime_event_history_buffer = validate_platform_runtime_event_history_buffer

__all__ = [
    "validate_platform_runtime_event_history_buffer",
    "validate_runtime_event_history_buffer",
]
