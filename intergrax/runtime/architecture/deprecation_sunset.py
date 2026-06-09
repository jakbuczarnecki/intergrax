# © Artur Czarnecki. All rights reserved.

"""Deprecation sunset window enforcement (IDEAL-31.5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState


@dataclass(frozen=True, slots=True)
class SunsetWindow:
    deprecated_at: datetime
    sunset_days: int = 90

    def sunset_at(self) -> datetime:
        return self.deprecated_at + timedelta(days=self.sunset_days)

    def is_past_sunset(self, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        return current >= self.sunset_at()


def should_block_deprecated_routing(
    lifecycle_state: AgentLifecycleState,
    window: SunsetWindow | None,
    *,
    now: datetime | None = None,
) -> bool:
    if lifecycle_state is not AgentLifecycleState.DEPRECATED:
        return False
    if window is None:
        return False
    return window.is_past_sunset(now=now)
