# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Output formatters for debug CLI."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunSummary,
    SerializedTraceEvent,
)


def format_run_list(runs: List[RunSummary]) -> str:
    if not runs:
        return "No runs found for tenant."
    lines = [
        f"{'RUN_ID':<36} {'STARTED':<26} {'EVENTS':>6} {'DURATION_MS':>12} {'USER':<12}",
        "-" * 96,
    ]
    for run in runs:
        lines.append(
            f"{run.run_id:<36} {run.started_at_utc:<26} {run.event_count:>6} "
            f"{run.duration_ms:>12} {run.user_id:<12}"
        )
    return "\n".join(lines)


def format_run_show(persisted: PersistedRun) -> str:
    meta = persisted.metadata
    llm_usage = dict(meta.stats.llm_usage or {})
    cost = llm_usage.get("cost")
    lines = [
        f"run_id:      {meta.run_id}",
        f"tenant_id:   {meta.tenant_id}",
        f"user_id:     {meta.user_id}",
        f"session_id:  {meta.session_id}",
        f"started_at:  {meta.started_at_utc}",
        f"duration_ms: {meta.stats.duration_ms}",
        f"cost:        {cost if cost is not None else '(none)'}",
        f"events:      {len(persisted.events)}",
    ]
    if meta.error is not None:
        lines.append(f"error:       {meta.error.error_type.value} — {meta.error.message}")
    return "\n".join(lines)


def _normalize_trace_event(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, SerializedTraceEvent):
        from dataclasses import asdict

        return asdict(raw)
    raise TypeError(f"Unsupported trace event type: {type(raw)!r}")


def build_trace_payload(
    persisted: PersistedRun,
    *,
    include_runtime: bool = False,
    runtime_store: RuntimeEventPersistence | None = None,
) -> Dict[str, Any]:
    trace_events = [_normalize_trace_event(raw) for raw in persisted.events]
    payload: Dict[str, Any] = {
        "run_id": persisted.metadata.run_id,
        "tenant_id": persisted.metadata.tenant_id,
        "trace_events": trace_events,
    }
    if not include_runtime or runtime_store is None:
        return payload

    from intergrax.runtime.events.unified_run_journal import (
        JOURNAL_SCHEMA_VERSION,
        build_unified_run_journal,
    )

    journal = build_unified_run_journal(persisted, runtime_store=runtime_store)
    payload["runtime_events"] = [event.model_dump(mode="json") for event in journal]
    payload["journal_schema_version"] = JOURNAL_SCHEMA_VERSION
    return payload


def format_trace_timeline(persisted: PersistedRun) -> str:
    lines = [f"Trace timeline for run {persisted.metadata.run_id} ({len(persisted.events)} events)", ""]
    for raw in persisted.events:
        if not isinstance(raw, dict):
            continue
        seq = raw.get("seq", "?")
        ts = raw.get("ts_utc", "")
        step = raw.get("step", "")
        message = raw.get("message", "")
        lines.append(f"[{seq:>3}] {ts}  {step:<20}  {message}")
    return "\n".join(lines)
