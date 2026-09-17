# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local RuntimeEventBus history contracts (OBS-RUNTIME-HISTORY-BOUNDS).

Local history is an ephemeral diagnostic projection — not canonical execution evidence.
Full historical reads must use ``EvidencePersistencePort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from intergrax.contracts.runtime_event import RuntimeEvent

DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY: int = 512


@dataclass(frozen=True, slots=True)
class RuntimeEventHistoryRetention:
    """Immutable retention declaration for a process-local history buffer."""

    mode: Literal["disabled", "bounded"]
    capacity: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "bounded":
            if self.capacity is None:
                raise ValueError("bounded retention requires capacity")
            if isinstance(self.capacity, bool) or type(self.capacity) is not int:
                raise ValueError("bounded retention capacity must be a positive int")
            if self.capacity <= 0:
                raise ValueError("bounded retention capacity must be > 0")
        elif self.capacity is not None:
            raise ValueError("disabled retention must not set capacity")


@runtime_checkable
class RuntimeEventHistoryBuffer(Protocol):
    """Platform process-local history storage (bounded deque or disabled).

    External plugins must not implement this as custom storage; use
    ``RuntimeEventHistoryStrategy`` instead.
    """

    def retention(self) -> RuntimeEventHistoryRetention: ...

    def append(self, event: RuntimeEvent) -> None: ...

    def snapshot(self) -> tuple[RuntimeEvent, ...]: ...

    def clear(self) -> None: ...


@runtime_checkable
class RuntimeEventHistoryStrategy(Protocol):
    """Stateless observer over the platform-owned bounded retention window."""

    def on_history_window(self, window: tuple[RuntimeEvent, ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeEventHistoryPolicy:
    """Explicit local history mode for ``RuntimeEventBus`` composition."""

    mode: Literal["disabled", "bounded"]
    max_events: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "bounded":
            if self.max_events is None:
                raise ValueError("bounded history requires max_events")
            if isinstance(self.max_events, bool) or type(self.max_events) is not int:
                raise ValueError("bounded history max_events must be a positive int")
            if self.max_events <= 0:
                raise ValueError("bounded history max_events must be > 0")
        elif self.max_events is not None:
            raise ValueError("disabled history must not set max_events")

    @classmethod
    def disabled(cls) -> RuntimeEventHistoryPolicy:
        return cls(mode="disabled", max_events=None)

    @classmethod
    def bounded(cls, max_events: int) -> RuntimeEventHistoryPolicy:
        if isinstance(max_events, bool) or type(max_events) is not int:
            raise ValueError("bounded history max_events must be a positive int")
        return cls(mode="bounded", max_events=max_events)

    @classmethod
    def enterprise_default(cls) -> RuntimeEventHistoryPolicy:
        return cls.bounded(DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY)
