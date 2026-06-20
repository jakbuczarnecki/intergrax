# © Artur Czarnecki. All rights reserved.

"""Security spine subscriber for ops counters (Phase SEC-ENT-5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.security.security_events import (
    KIND_DEFENSE_BLOCKED,
    KIND_ENCRYPTION_DENIED,
)


@dataclass
class SecuritySpineCounters:
    """In-process counters for platform.security.* domain signals."""

    defense_blocked: int = 0
    encryption_denied: int = 0
    subscription_id: str = ""
    _seen_event_ids: set[str] = field(default_factory=set, repr=False)

    def _count_once(self, event: RuntimeEvent, *, kind: str) -> None:
        event_id = event.event_id or ""
        if event_id and event_id in self._seen_event_ids:
            return
        if event_id:
            self._seen_event_ids.add(event_id)
        if kind == KIND_DEFENSE_BLOCKED:
            self.defense_blocked += 1
        elif kind == KIND_ENCRYPTION_DENIED:
            self.encryption_denied += 1


def wire_security_spine_subscriber(event_bus: object) -> SecuritySpineCounters:
    """Subscribe to ``platform.security.*`` kinds and increment counters."""
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    if not isinstance(event_bus, RuntimeEventBus):
        return SecuritySpineCounters()
    counters = SecuritySpineCounters()

    def _handler(event: RuntimeEvent) -> None:
        kind = event.event_kind or ""
        if kind in {KIND_DEFENSE_BLOCKED, KIND_ENCRYPTION_DENIED}:
            counters._count_once(event, kind=kind)

    event_bus.unsubscribe("platform.security.counters")
    counters.subscription_id = event_bus.subscribe(
        _handler,
        kind_prefix="platform.security.",
        priority=200,
        subscription_id="platform.security.counters",
    )
    return counters
