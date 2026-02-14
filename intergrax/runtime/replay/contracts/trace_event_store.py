# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol, Iterable

from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO


class TraceEventStore(Protocol):
    """Read-only access to trace events."""

    def get_events(self, run_id: str) -> Iterable[TraceEventDTO]:
        ...
