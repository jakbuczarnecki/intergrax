# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference in-memory capability usage consumer (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

import threading

from intergrax.capability_metering.errors import CapabilityUsageConflictError
from intergrax.contracts.capability_metering import CapabilityUsageEvent
from intergrax.contracts.execution_identity import EventId


class InMemoryCapabilityUsageConsumer:
    """Append-only reference consumer preserving first-seen order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[CapabilityUsageEvent] = []
        self._events_by_id: dict[EventId, CapabilityUsageEvent] = {}

    def consume(self, event: CapabilityUsageEvent) -> None:
        with self._lock:
            existing = self._events_by_id.get(event.event_id)
            if existing is not None:
                if existing == event:
                    return
                raise CapabilityUsageConflictError(
                    f"conflicting capability usage event for event_id={event.event_id!s}",
                )
            self._events_by_id[event.event_id] = event
            self._events.append(event)

    def snapshot(self) -> tuple[CapabilityUsageEvent, ...]:
        with self._lock:
            return tuple(self._events)
