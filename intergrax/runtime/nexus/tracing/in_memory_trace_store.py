# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List

from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunSummary,
    RunTraceReader,
    RunTraceWriter,
    SerializedTraceEvent,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent


class InMemoryRunTraceStore(RunTraceWriter, RunTraceReader):
    """
    Minimal, deterministic trace store for tests and local runs.
    Stores serialized TraceEvents per run_id.
    """

    def __init__(self) -> None:
        self._events_by_run: Dict[str, List[SerializedTraceEvent]] = {}
        self._metadata_by_run: Dict[str, RunMetadata] = {}

    def append_event(self, event: TraceEvent) -> None:
        run_id = event.run_id
        if run_id not in self._events_by_run:
            self._events_by_run[run_id] = []
        
        self._events_by_run[run_id].append(
            SerializedTraceEvent.from_trace_event(event)
        )

    def finalize_run(self, run_id: str, metadata: RunMetadata) -> None:
        from intergrax.runtime.nexus.tracing.parser_trace_flush import export_parser_traces_from_events

        export_parser_traces_from_events(self._events_by_run.get(run_id, []))
        self._metadata_by_run[run_id] = metadata

    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        if run_id not in self._events_by_run:
            raise KeyError(f"Run '{run_id}' not found in trace store")

        metadata = self._metadata_by_run.get(run_id)
        if metadata is None:
            raise KeyError(
                f"Run '{run_id}' metadata not found in trace store. Did you forget to call finalize_run()?"                
            )
        
        if metadata.tenant_id != tenant_id:
            raise KeyError(f"Run '{run_id}' not found in trace store")

        return PersistedRun(
            metadata=metadata,
            events=list(self._events_by_run[run_id]),
        )

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> List[RunSummary]:
        rows: List[RunSummary] = []
        for run_id, metadata in self._metadata_by_run.items():
            if metadata.tenant_id != tenant_id:
                continue
            rows.append(
                RunSummary(
                    run_id=run_id,
                    tenant_id=metadata.tenant_id,
                    user_id=metadata.user_id,
                    session_id=metadata.session_id,
                    started_at_utc=metadata.started_at_utc,
                    duration_ms=metadata.stats.duration_ms,
                    event_count=len(self._events_by_run.get(run_id, [])),
                )
            )
        rows.sort(key=lambda r: r.started_at_utc, reverse=True)
        return rows[:limit]
