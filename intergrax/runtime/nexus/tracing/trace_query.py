# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Type, TypeVar

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceEvent

T = TypeVar("T", bound=DiagnosticPayload)


@dataclass(frozen=True)
class TraceQuery:
    """
    Query helper for TraceEvent collections.

    Goals:
    - typed-first access (payload instances, not dicts)
    - no assumptions about debug_trace dict structure
    - small, stable API for notebooks/tests/diagnostics
    """

    events: Sequence[TraceEvent]

    @staticmethod
    def from_iter(events: Iterable[TraceEvent]) -> "TraceQuery":
        return TraceQuery(events=list(events))

    def all_events(self) -> List[TraceEvent]:
        return list(self.events)

    def all_payloads(self, payload_type: Type[T]) -> List[T]:
        out: List[T] = []
        for e in self.events:
            p = e.payload
            if isinstance(p, payload_type):
                out.append(p)
        return out

    def first_payload(self, payload_type: Type[T]) -> Optional[T]:
        for e in self.events:
            p = e.payload
            if isinstance(p, payload_type):
                return p
        return None

    def one_payload(self, payload_type: Type[T]) -> T:
        items = self.all_payloads(payload_type)
        if len(items) != 1:
            raise ValueError(
                f"Expected exactly 1 payload of type {payload_type.__name__}, got {len(items)}"
            )
        return items[0]

    def all_events_with_payload(self, payload_type: Type[T]) -> List[TraceEvent]:
        out: List[TraceEvent] = []
        for e in self.events:
            if isinstance(e.payload, payload_type):
                out.append(e)
        return out

    def first_event_with_payload(self, payload_type: Type[T]) -> Optional[TraceEvent]:
        for e in self.events:
            if isinstance(e.payload, payload_type):
                return e
        return None

    def one_event_with_payload(self, payload_type: Type[T]) -> TraceEvent:
        items = self.all_events_with_payload(payload_type)
        if len(items) != 1:
            raise ValueError(
                f"Expected exactly 1 event with payload type {payload_type.__name__}, got {len(items)}"
            )
        return items[0]
