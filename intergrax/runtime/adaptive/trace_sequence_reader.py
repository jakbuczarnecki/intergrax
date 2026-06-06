# © Artur Czarnecki. All rights reserved.

"""Trace sequence reader for process pattern mining (Phase W-ADAPT-6.2)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.signal_store import SignalStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunTraceReader


class ProcessSequenceToken(BaseModel):
    """Single step token in a mined process sequence (AHIA §9.7)."""

    model_config = ConfigDict(extra="forbid")

    task_class: str
    agent_id: str
    tool_id: str
    hitl_pause: bool = False
    outcome_success: bool = True


class RunProcessSequence(BaseModel):
    """Ordered process tokens extracted from one persisted run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tenant_id: str
    tokens: list[ProcessSequenceToken] = Field(default_factory=list)
    utility: float | None = None


class TraceSequenceReader(Protocol):
    """Loads tenant-scoped run sequences for offline pattern mining."""

    def load_sequences(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[RunProcessSequence]:
        ...


class PersistedTraceSequenceReader:
    """Build run sequences from ``RunTraceReader`` and optional ``SignalStore``."""

    def __init__(
        self,
        trace_reader: RunTraceReader,
        *,
        signal_store: SignalStore | None = None,
    ) -> None:
        self._trace_reader = trace_reader
        self._signal_store = signal_store

    def load_sequences(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[RunProcessSequence]:
        summaries = self._trace_reader.list_runs(tenant_id, limit=limit)
        utility_by_run = self._utility_index(tenant_id=tenant_id)
        sequences: list[RunProcessSequence] = []
        for summary in summaries:
            persisted = self._trace_reader.read_run(summary.run_id, tenant_id)
            sequences.append(
                RunProcessSequence(
                    run_id=summary.run_id,
                    tenant_id=tenant_id,
                    tokens=extract_tokens_from_run(persisted),
                    utility=utility_by_run.get(summary.run_id),
                )
            )
        return sequences

    def _utility_index(self, *, tenant_id: str) -> dict[str, float]:
        if self._signal_store is None:
            return {}
        signals = self._signal_store.list_signals(tenant_id=tenant_id, limit=5000)
        index: dict[str, float] = {}
        for signal in signals:
            if signal.utility is None:
                continue
            index[signal.run_id] = signal.utility
        return index


def extract_tokens_from_run(persisted: PersistedRun) -> list[ProcessSequenceToken]:
    """Extract n-gram tokens from serialized trace events."""
    outcome_success = persisted.metadata.error is None
    default_task_class = "unknown"
    default_agent_id = persisted.metadata.user_id or "unknown"
    tokens: list[ProcessSequenceToken] = []
    hitl_seen = False

    for raw in persisted.events:
        event = _coerce_event_dict(raw)
        tags = dict(event.get("tags") or {})
        step = str(event.get("step", ""))
        component = str(event.get("component", ""))
        task_class = str(tags.get("capability") or tags.get("task_class") or default_task_class)
        agent_id = str(tags.get("agent_id") or default_agent_id)
        if "hitl" in step.lower() or "human" in step.lower():
            hitl_seen = True
        tool_id = _resolve_tool_id(step=step, component=component, tags=tags)
        if tool_id:
            tokens.append(
                ProcessSequenceToken(
                    task_class=task_class,
                    agent_id=agent_id,
                    tool_id=tool_id,
                    hitl_pause=hitl_seen,
                    outcome_success=outcome_success,
                )
            )

    if not tokens:
        tokens.append(
            ProcessSequenceToken(
                task_class=default_task_class,
                agent_id=default_agent_id,
                tool_id="_none_",
                hitl_pause=hitl_seen,
                outcome_success=outcome_success,
            )
        )
    return tokens


def _coerce_event_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if is_dataclass(raw):
        return asdict(raw)
    return {}


def _resolve_tool_id(*, step: str, component: str, tags: dict[str, Any]) -> str:
    explicit = tags.get("tool_id")
    if explicit:
        return str(explicit)
    lowered_step = step.lower()
    if "tool" in lowered_step or component.lower() in {"tool_runtime", "tools"}:
        return step or component or "tool"
    return ""
