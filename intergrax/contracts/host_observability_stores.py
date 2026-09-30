# © Artur Czarnecki. All rights reserved.

"""Neutral observability store bundle for Tier-3 host composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.run_trace_store import RunTraceStore


@dataclass(frozen=True)
class HostObservabilityStores:
    """Trace + runtime event backends for application composition roots."""

    trace_store: RunTraceStore
    runtime_event_store: object | None
    trace_db_path: Path | None
    runtime_events_db_path: Path | None


__all__ = ["HostObservabilityStores"]
