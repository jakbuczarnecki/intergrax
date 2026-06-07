# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified run journal — merges persisted Nexus trace with ``RuntimeEvent`` store (§42.24).

Trace replay and live ``RuntimeEventBus`` persistence are dual paths; this module
produces one chronological ``RuntimeEvent`` timeline for operators and debug tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, SerializedTraceEvent
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

JOURNAL_SCHEMA_VERSION = "unified_run_journal.v1"


class _TaskLike(Protocol):
    task_id: str
    tenant_id: str
    agent_id: str | None


def build_unified_run_journal(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence | None = None,
    limit: int = 2000,
) -> List[RuntimeEvent]:
    """
    Merge persisted runtime events with trace-bridged events for one run.

    Persisted bus events win on ``event_id`` / ``trace_event_id`` collisions.
    When ``runtime_store`` is ``None``, returns trace-bridge view only.
    """
    tenant_id = persisted.metadata.tenant_id
    run_id = persisted.metadata.run_id
    bridged = bridge_persisted_trace_events(persisted)

    if runtime_store is None:
        return _sort_journal(bridged)[:limit]

    stored = runtime_store.list_for_run(run_id, tenant_id=tenant_id, limit=limit)
    if not stored:
        return _sort_journal(bridged)[:limit]

    return _merge_journal(stored, bridged, limit=limit)


def bridge_persisted_trace_events(persisted: PersistedRun) -> List[RuntimeEvent]:
    """Map all persisted trace rows to canonical ``RuntimeEvent`` via ``trace_bridge``."""
    tenant_id = persisted.metadata.tenant_id
    run_id = persisted.metadata.run_id
    events: List[RuntimeEvent] = []
    task = _task_from_tags({}, run_id=run_id, tenant_id=tenant_id)

    for raw in persisted.events:
        normalized = _normalize_trace_row(raw)
        tags = normalized.get("tags") or {}
        task = _task_from_tags(
            {**tags, "tenant_id": tenant_id},
            run_id=run_id,
            tenant_id=tenant_id,
        )
        trace = _trace_event_from_dict(normalized, default_run_id=run_id)
        events.append(
            trace_event_to_runtime_event(
                trace,
                task,  # type: ignore[arg-type]
                payload_schema_id=_optional_str(normalized.get("payload_schema_id")),
                payload_dict=_payload_dict(normalized.get("payload")),
            )
        )
    return events


def _merge_journal(
    persisted: List[RuntimeEvent],
    bridged: List[RuntimeEvent],
    *,
    limit: int,
) -> List[RuntimeEvent]:
    persisted_ids = {event.event_id for event in persisted}
    covered_trace_ids = {
        str(event.payload["trace_event_id"])
        for event in persisted
        if event.payload.get("trace_event_id")
    }

    merged = list(persisted)
    for event in bridged:
        trace_id = str(event.payload.get("trace_event_id", ""))
        if event.event_id in persisted_ids:
            continue
        if trace_id and trace_id in covered_trace_ids:
            continue
        merged.append(event)

    return _sort_journal(merged)[:limit]


def _sort_journal(events: List[RuntimeEvent]) -> List[RuntimeEvent]:
    return sorted(events, key=_journal_sort_key)


def _journal_sort_key(event: RuntimeEvent) -> tuple[Any, ...]:
    seq = event.payload.get("trace_seq", 0)
    try:
        seq_num = int(seq)
    except (TypeError, ValueError):
        seq_num = 0
    return (event.timestamp, seq_num, event.event_id)


def _task_from_tags(tags: Mapping[str, Any], *, run_id: str, tenant_id: str) -> _TaskLike:
    return _TaskView(
        task_id=str(tags.get("task_id") or run_id),
        tenant_id=str(tags.get("tenant_id") or tenant_id or "default"),
        agent_id=tags.get("agent_id") if tags.get("agent_id") is not None else None,
        capability=tags.get("capability"),
    )


class _TaskView:
    __slots__ = ("task_id", "tenant_id", "agent_id", "context")

    def __init__(
        self,
        task_id: str,
        tenant_id: str,
        agent_id: str | None,
        capability: Any,
    ) -> None:
        self.task_id = task_id
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.context = type("Ctx", (), {"capability": capability})()


def _normalize_trace_row(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, SerializedTraceEvent):
        from dataclasses import asdict

        return asdict(raw)
    raise TypeError(f"Unsupported trace event type: {type(raw)!r}")


def _trace_event_from_dict(raw: Mapping[str, Any], *, default_run_id: str) -> TraceEvent:
    tags = dict(raw.get("tags") or {})
    return TraceEvent(
        event_id=str(raw.get("event_id", "")),
        run_id=str(raw.get("run_id", default_run_id)),
        seq=int(raw.get("seq", 0)),
        ts_utc=str(raw.get("ts_utc", "")),
        level=TraceLevel(str(raw.get("level", "info"))),
        component=TraceComponent(str(raw.get("component", "planner"))),
        step=str(raw.get("step", "")),
        message=str(raw.get("message", "")),
        tags=tags,
    )


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _payload_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    return None
