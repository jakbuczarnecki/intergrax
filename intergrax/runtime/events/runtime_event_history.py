# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default process-local RuntimeEventBus history buffer implementations."""

from __future__ import annotations

import threading
from collections import deque

from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_history import (
    DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY,
    RuntimeEventHistoryBuffer,
    RuntimeEventHistoryPolicy,
)


class DisabledRuntimeEventHistory:
    """Null-object history buffer (no retention)."""

    __slots__ = ()

    def append(self, event: RuntimeEvent) -> None:
        del event

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return ()

    def clear(self) -> None:
        return


class BoundedRuntimeEventHistory:
    """Fixed-capacity ring buffer; evicts oldest on overflow."""

    __slots__ = ("_capacity", "_deque", "_lock")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._deque: deque[RuntimeEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._deque.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._deque)

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()


def runtime_event_history_buffer_from_policy(
    policy: RuntimeEventHistoryPolicy,
) -> RuntimeEventHistoryBuffer:
    if policy.mode == "disabled":
        return DisabledRuntimeEventHistory()
    assert policy.max_events is not None
    return BoundedRuntimeEventHistory(policy.max_events)


def resolve_runtime_event_history_buffer(
    *,
    record_history: bool | None,
    history_policy: RuntimeEventHistoryPolicy | None,
    history_buffer: RuntimeEventHistoryBuffer | None,
) -> RuntimeEventHistoryBuffer:
    if history_buffer is not None:
        if record_history is not None or history_policy is not None:
            raise ValueError(
                "history_buffer cannot be combined with record_history or history_policy",
            )
        return history_buffer
    if record_history is not None and history_policy is not None:
        raise ValueError("record_history and history_policy are mutually exclusive")
    if record_history is False:
        return DisabledRuntimeEventHistory()
    if history_policy is not None:
        return runtime_event_history_buffer_from_policy(history_policy)
    if record_history is True:
        return BoundedRuntimeEventHistory(
            DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY
        )
    return runtime_event_history_buffer_from_policy(
        RuntimeEventHistoryPolicy.enterprise_default()
    )


__all__ = [
    "BoundedRuntimeEventHistory",
    "DisabledRuntimeEventHistory",
    "DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY",
    "resolve_runtime_event_history_buffer",
    "runtime_event_history_buffer_from_policy",
]
