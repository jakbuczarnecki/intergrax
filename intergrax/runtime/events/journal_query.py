# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Journal read-model filters by taxonomy (OBS-EVOL-9.5 · SAR-07)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.contracts.execution_phase import ExecutionPhase
    from intergrax.runtime.events.event_taxonomy import EventCategory
    from intergrax.runtime.events.runtime_event import RuntimeEvent


def query_journal(
    events: Sequence[RuntimeEvent],
    *,
    categories: set[EventCategory] | None = None,
    kind_prefix: str | None = None,
    ops_hints: set[str] | None = None,
    phase: ExecutionPhase | None = None,
) -> list[RuntimeEvent]:
    """Filter a journal timeline by derived taxonomy fields."""
    if not any((categories, kind_prefix, ops_hints, phase is not None)):
        return list(events)
    result: list[RuntimeEvent] = []
    for event in events:
        if categories is not None:
            if event.event_category is None or event.event_category not in categories:
                continue
        if kind_prefix is not None and not (event.event_kind or "").startswith(kind_prefix):
            continue
        if ops_hints is not None and event.ops_hint not in ops_hints:
            continue
        if phase is not None and event.phase != phase:
            continue
        result.append(event)
    return result
