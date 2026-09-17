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
    RuntimeEventHistoryRetention,
)
from intergrax.runtime.events.runtime_event_history_validation import (
    validate_runtime_event_history_buffer,
)


class PlatformOwnedRuntimeEventHistoryBuffer:
    """Platform retention envelope; optional custom strategy stays within the window."""

    __slots__ = ("_deque", "_lock", "_mode", "_strategy")

    def __init__(
        self,
        retention: RuntimeEventHistoryRetention,
        *,
        strategy: RuntimeEventHistoryBuffer | None = None,
    ) -> None:
        self._mode = retention.mode
        self._strategy = strategy
        self._lock = threading.Lock()
        if retention.mode == "disabled":
            self._deque: deque[RuntimeEvent] | None = None
        else:
            assert retention.capacity is not None
            self._deque = deque(maxlen=retention.capacity)

    def retention(self) -> RuntimeEventHistoryRetention:
        if self._mode == "disabled":
            return RuntimeEventHistoryRetention(mode="disabled", capacity=None)
        assert self._deque is not None
        capacity = self._deque.maxlen
        assert capacity is not None
        return RuntimeEventHistoryRetention(mode="bounded", capacity=capacity)

    def append(self, event: RuntimeEvent) -> None:
        if self._mode == "disabled":
            del event
            return
        assert self._deque is not None
        with self._lock:
            self._deque.append(event)
            self._sync_strategy()

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        if self._mode == "disabled":
            return ()
        assert self._deque is not None
        with self._lock:
            return tuple(self._deque)

    def clear(self) -> None:
        with self._lock:
            if self._deque is not None:
                self._deque.clear()
            if self._strategy is not None:
                self._strategy.clear()

    def _sync_strategy(self) -> None:
        strategy = self._strategy
        if strategy is None or self._deque is None:
            return
        window = tuple(self._deque)
        strategy.clear()
        for item in window:
            strategy.append(item)


class DisabledRuntimeEventHistory:
    """Null-object history buffer (no retention)."""

    __slots__ = ()

    def retention(self) -> RuntimeEventHistoryRetention:
        return RuntimeEventHistoryRetention(mode="disabled", capacity=None)

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
        if isinstance(capacity, bool) or type(capacity) is not int:
            raise ValueError("capacity must be a positive int")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._deque: deque[RuntimeEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def retention(self) -> RuntimeEventHistoryRetention:
        return RuntimeEventHistoryRetention(mode="bounded", capacity=self._capacity)

    def append(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._deque.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._deque)

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()


def _retention_from_policy(
    policy: RuntimeEventHistoryPolicy,
) -> RuntimeEventHistoryRetention:
    if policy.mode == "disabled":
        return RuntimeEventHistoryRetention(mode="disabled", capacity=None)
    assert policy.max_events is not None
    return RuntimeEventHistoryRetention(mode="bounded", capacity=policy.max_events)


def runtime_event_history_buffer_from_policy(
    policy: RuntimeEventHistoryPolicy,
) -> RuntimeEventHistoryBuffer:
    return PlatformOwnedRuntimeEventHistoryBuffer(_retention_from_policy(policy))


def wrap_runtime_event_history_strategy(
    strategy: RuntimeEventHistoryBuffer,
) -> PlatformOwnedRuntimeEventHistoryBuffer:
    validate_runtime_event_history_buffer(strategy)
    retention = strategy.retention()
    if retention.mode == "disabled":
        raise ValueError("custom history strategy cannot be disabled when injected")
    return PlatformOwnedRuntimeEventHistoryBuffer(retention, strategy=strategy)


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
        return wrap_runtime_event_history_strategy(history_buffer)
    if record_history is not None and history_policy is not None:
        raise ValueError("record_history and history_policy are mutually exclusive")
    if record_history is False:
        buffer = PlatformOwnedRuntimeEventHistoryBuffer(
            RuntimeEventHistoryRetention(mode="disabled", capacity=None),
        )
    elif history_policy is not None:
        buffer = runtime_event_history_buffer_from_policy(history_policy)
    elif record_history is True:
        buffer = PlatformOwnedRuntimeEventHistoryBuffer(
            RuntimeEventHistoryRetention(
                mode="bounded",
                capacity=DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY,
            ),
        )
    else:
        buffer = runtime_event_history_buffer_from_policy(
            RuntimeEventHistoryPolicy.enterprise_default()
        )
    validate_runtime_event_history_buffer(buffer)
    return buffer


__all__ = [
    "BoundedRuntimeEventHistory",
    "DisabledRuntimeEventHistory",
    "PlatformOwnedRuntimeEventHistoryBuffer",
    "DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY",
    "resolve_runtime_event_history_buffer",
    "runtime_event_history_buffer_from_policy",
    "wrap_runtime_event_history_strategy",
]
