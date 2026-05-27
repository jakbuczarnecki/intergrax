# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Output formatters for debug CLI."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunSummary


class _TaskView:
    """Minimal duck type for trace_bridge (avoids heavy task package import)."""

    __slots__ = ("task_id", "agent_id", "context")

    def __init__(self, task_id: str, agent_id: str | None, capability: Any) -> None:
        self.task_id = task_id
        self.agent_id = agent_id
        self.context = type("Ctx", (), {"capability": capability})()


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


def _task_from_trace_tags(tags: Dict[str, Any], run_id: str) -> _TaskView:
    return _TaskView(
        task_id=str(tags.get("task_id") or run_id),
        agent_id=tags.get("agent_id"),
        capability=tags.get("capability"),
    )


def build_trace_payload(
    persisted: PersistedRun,
    *,
    include_runtime: bool = False,
) -> Dict[str, Any]:
    trace_events = persisted.events
    payload: Dict[str, Any] = {
        "run_id": persisted.metadata.run_id,
        "tenant_id": persisted.metadata.tenant_id,
        "trace_events": trace_events,
    }
    if not include_runtime:
        return payload

    from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
    from intergrax.runtime.nexus.tracing.trace_models import (
        TraceComponent,
        TraceEvent,
        TraceLevel,
    )

    runtime_events: List[Dict[str, Any]] = []
    task = _task_from_trace_tags({}, persisted.metadata.run_id)
    for raw in trace_events:
        if not isinstance(raw, dict):
            continue
        tags = raw.get("tags") or {}
        task = _task_from_trace_tags(
            {
                **tags,
                "tenant_id": persisted.metadata.tenant_id,
                "user_id": persisted.metadata.user_id,
            },
            persisted.metadata.run_id,
        )
        trace = TraceEvent(
            event_id=str(raw.get("event_id", "")),
            run_id=str(raw.get("run_id", persisted.metadata.run_id)),
            seq=int(raw.get("seq", 0)),
            ts_utc=str(raw.get("ts_utc", "")),
            level=TraceLevel(str(raw.get("level", "info"))),
            component=TraceComponent(str(raw.get("component", "planner"))),
            step=str(raw.get("step", "")),
            message=str(raw.get("message", "")),
            tags=tags,
        )
        runtime_events.append(
            trace_event_to_runtime_event(trace, task).model_dump(mode="json")
        )
    payload["runtime_events"] = runtime_events
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
