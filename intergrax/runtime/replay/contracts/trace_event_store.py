# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable

from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO


class TraceEventStore(ABC):
    """Read-only access to trace events."""

    @abstractmethod
    def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
        ...
