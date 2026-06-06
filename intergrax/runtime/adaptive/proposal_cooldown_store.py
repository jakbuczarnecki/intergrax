# © Artur Czarnecki. All rights reserved.

"""Proposal cooldown tracking (Phase W-ADAPT-2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class ProposalCooldownStore(Protocol):
    """Tracks last proposal timestamps per loop_id."""

    def is_on_cooldown(self, loop_id: str, *, cooldown_seconds: int, now: datetime | None = None) -> bool: ...

    def mark_proposed(self, loop_id: str, *, now: datetime | None = None) -> None: ...

    def clear(self) -> None: ...


class InMemoryProposalCooldownStore:
    """In-process cooldown tracker."""

    def __init__(self) -> None:
        self._last_proposed: dict[str, datetime] = {}

    def is_on_cooldown(
        self,
        loop_id: str,
        *,
        cooldown_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        last = self._last_proposed.get(loop_id)
        if last is None:
            return False
        current = now or datetime.now(UTC)
        elapsed = (current - last).total_seconds()
        return elapsed < cooldown_seconds

    def mark_proposed(self, loop_id: str, *, now: datetime | None = None) -> None:
        self._last_proposed[loop_id] = now or datetime.now(UTC)

    def clear(self) -> None:
        self._last_proposed.clear()
