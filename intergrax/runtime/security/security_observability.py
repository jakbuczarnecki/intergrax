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


def wire_security_spine_subscriber(event_bus: object) -> SecuritySpineCounters:
    """Subscribe to ``platform.security.*`` kinds and increment counters."""
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    if not isinstance(event_bus, RuntimeEventBus):
        return SecuritySpineCounters()
    counters = SecuritySpineCounters()

    def _handler(event: RuntimeEvent) -> None:
        kind = event.event_kind or ""
        if kind == KIND_DEFENSE_BLOCKED:
            counters.defense_blocked += 1
        elif kind == KIND_ENCRYPTION_DENIED:
            counters.encryption_denied += 1

    counters.subscription_id = event_bus.subscribe(
        _handler,
        kind_prefix="platform.security.",
        priority=200,
        subscription_id="platform.security.counters",
    )
    return counters
