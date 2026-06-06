# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Iterable, List

from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader, SerializedTraceEvent
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
from intergrax.runtime.replay.contracts.trace_event_store import TraceEventStore
from intergrax.runtime.replay.trace_replay_bridge import serialized_trace_events_to_replay_dtos


class PersistedRunTraceEventStore(TraceEventStore):
    """Read replay DTOs from a persisted Nexus run trace."""

    def __init__(self, reader: RunTraceReader) -> None:
        self._reader = reader

    def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
        persisted = self._reader.read_run(run_id, tenant_id)
        serialized: List[SerializedTraceEvent] = []
        for raw in persisted.events:
            if isinstance(raw, SerializedTraceEvent):
                serialized.append(raw)
            elif isinstance(raw, dict):
                serialized.append(
                    SerializedTraceEvent(
                        event_id=str(raw.get("event_id", "")),
                        run_id=str(raw.get("run_id", run_id)),
                        seq=int(raw.get("seq", 0) or 0),
                        ts_utc=str(raw.get("ts_utc", "")),
                        level=str(raw.get("level", "info")),
                        component=str(raw.get("component", "")),
                        step=str(raw.get("step", "")),
                        message=str(raw.get("message", "")),
                        payload_schema_id=raw.get("payload_schema_id"),
                        payload_schema_version=raw.get("payload_schema_version"),
                        payload=raw.get("payload"),
                        tags=dict(raw.get("tags") or {}),
                        artifact_refs=[],
                    )
                )
        return serialized_trace_events_to_replay_dtos(serialized)
