# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified run metrics export (Appendix B.11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from intergrax.runtime.governance.contracts.metrics_record_dto import RunMetricsRecord
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, SerializedTraceEvent
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent
from intergrax.runtime.observability.modality_metrics import ModalityMetricsPayload
from intergrax.runtime.replay.metrics import ExecutionMetrics

_LLM_SCHEMA_MARKERS = frozenset({"llm", "llm_usage", "llm_call"})
_TOOL_SCHEMA_MARKERS = frozenset({"tool", "tool_call"})
_GRAPH_MESSAGE_MARKERS = frozenset({"graph"})


@dataclass(slots=True)
class RunMetricsExport:
    run_id: str
    tenant_id: str
    agent_id: Optional[str]
    duration_ms: int
    event_count: int
    cost: Optional[float] = None
    total_tokens: Optional[int] = None
    llm_usage: Dict[str, Any] = field(default_factory=dict)
    trace_summary: Dict[str, int] = field(default_factory=dict)
    modality_metrics: ModalityMetricsPayload = field(default_factory=ModalityMetricsPayload)
    behavioral: Optional[ExecutionMetrics] = None


def export_run_metrics(persisted: PersistedRun, *, agent_id: Optional[str] = None) -> RunMetricsExport:
    meta = persisted.metadata
    llm_usage = dict(meta.stats.llm_usage or {})
    trace_summary = _summarize_trace_events(persisted.events)
    behavioral: ExecutionMetrics | None = None
    if persisted.events:
        step_count = max(trace_summary.get("step_events", 0), 1)
        behavioral = ExecutionMetrics(
            step_count=step_count,
            total_llm_calls=trace_summary.get("llm_events", 0),
            total_tool_calls=trace_summary.get("tool_events", 0),
            total_artifacts=0,
            total_tokens=_coerce_int(llm_usage.get("total_tokens")) or 0,
            duration=meta.stats.duration_ms / 1000.0 if meta.stats.duration_ms else None,
            tool_steps_ratio=min(1.0, trace_summary.get("tool_events", 0) / step_count),
            llm_steps_ratio=min(1.0, trace_summary.get("llm_events", 0) / step_count),
        )

    modality_metrics = _extract_modality_metrics_from_trace(persisted.events)
    return RunMetricsExport(
        run_id=meta.run_id,
        tenant_id=meta.tenant_id,
        agent_id=agent_id,
        duration_ms=meta.stats.duration_ms,
        event_count=len(persisted.events),
        cost=_coerce_float(llm_usage.get("cost")),
        total_tokens=_coerce_int(llm_usage.get("total_tokens")),
        llm_usage=llm_usage,
        trace_summary=trace_summary,
        modality_metrics=modality_metrics,
        behavioral=behavioral,
    )


def _extract_modality_metrics_from_trace(events: List[SerializedTraceEvent]) -> ModalityMetricsPayload:
    for event in reversed(events):
        payload = _trace_event_payload(event)
        if "modality_metrics" in payload:
            return ModalityMetricsPayload.model_validate(payload["modality_metrics"])
    return ModalityMetricsPayload()


def _trace_event_payload(event: SerializedTraceEvent | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(event, dict):
        raw = event.get("payload")
    else:
        raw = event.payload
    return raw if isinstance(raw, dict) else {}


def persist_run_metrics(
    *,
    store: ExecutionMetricsStore,
    persisted: PersistedRun,
    agent_id: str,
) -> RunMetricsExport:
    exported = export_run_metrics(persisted, agent_id=agent_id)
    behavioral = exported.behavioral
    if behavioral is None and exported.event_count > 0:
        summary = exported.trace_summary
        step_count = max(summary.get("step_events", 0), 1)
        behavioral = ExecutionMetrics(
            step_count=step_count,
            total_llm_calls=summary.get("llm_events", 0),
            total_tool_calls=summary.get("tool_events", 0),
            total_artifacts=0,
            total_tokens=exported.total_tokens or 0,
            duration=exported.duration_ms / 1000.0 if exported.duration_ms else None,
            tool_steps_ratio=min(1.0, summary.get("tool_events", 0) / step_count),
            llm_steps_ratio=min(1.0, summary.get("llm_events", 0) / step_count),
        )
        exported.behavioral = behavioral
    if behavioral is not None:
        store.save(
            RunMetricsRecord(
                run_id=exported.run_id,
                agent_id=agent_id,
                metrics=behavioral,
            )
        )
    return exported


def list_agent_metrics(
    store: ExecutionMetricsStore,
    agent_id: str,
    *,
    limit: int = 20,
) -> List[RunMetricsRecord]:
    return store.get_recent(agent_id, limit=limit)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _schema_id_from_serialized(event: SerializedTraceEvent) -> str:
    return str(event.payload_schema_id or "")


def _schema_id_from_event(event: TraceEvent | dict[str, Any]) -> str:
    if isinstance(event, dict):
        return str(event.get("payload_schema_id") or "")
    if event.payload is not None:
        return event.payload.__class__.schema_id()
    return ""


def _summarize_trace_events(
    events: List[TraceEvent | SerializedTraceEvent | dict[str, Any]],
) -> Dict[str, int]:
    graph_events = 0
    step_events = 0
    tool_events = 0
    llm_events = 0
    for event in events:
        if isinstance(event, dict):
            message = str(event.get("message") or "")
            step = str(event.get("step") or "")
            schema_id = _schema_id_from_event(event)
            component = str(event.get("component") or "")
        elif isinstance(event, TraceEvent):
            message = event.message
            step = event.step
            schema_id = _schema_id_from_event(event)
            component = event.component.value
        elif isinstance(event, SerializedTraceEvent):
            message = event.message
            step = event.step
            schema_id = _schema_id_from_serialized(event)
            component = event.component
        else:
            continue
        lower_msg = message.lower()
        if any(m in lower_msg for m in _GRAPH_MESSAGE_MARKERS):
            graph_events += 1
        if step:
            step_events += 1
        sid = schema_id.lower()
        if sid and any(m in sid for m in _TOOL_SCHEMA_MARKERS):
            tool_events += 1
        elif component in (TraceComponent.TOOLS.value, TraceComponent.STEP.value):
            if component == TraceComponent.TOOLS.value:
                tool_events += 1
        if sid and any(m in sid for m in _LLM_SCHEMA_MARKERS):
            llm_events += 1
        elif component in (TraceComponent.ENGINE.value, TraceComponent.STEP.value):
            if "llm" in step.lower():
                llm_events += 1
    return {
        "graph_events": graph_events,
        "step_events": step_events,
        "tool_events": tool_events,
        "llm_events": llm_events,
    }
