# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunError,
    RunMetadata,
    RunStats,
    RunSummary,
)
from intergrax.tools.providers.harness.contracts import (
    HarnessGetRunCostInput,
    HarnessGetRunEventsInput,
    HarnessGetRunInput,
    HarnessListRunsInput,
)
from intergrax.tools.providers.harness.service import (
    harness_get_run,
    harness_get_run_cost,
    harness_get_run_events,
    harness_list_runs,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryTraceReader:
    def __init__(self) -> None:
        self._metadata = RunMetadata(
            run_id="run-1",
            session_id="sess-1",
            user_id="u-1",
            tenant_id="tenant-a",
            started_at_utc="2026-06-07T10:00:00Z",
            stats=RunStats(duration_ms=120, llm_usage={"input_tokens": 10, "output_tokens": 5}),
            error=RunError(error_type=RuntimeErrorCode.INTERNAL_ERROR, message="boom"),
        )
        self._events = [
            {
                "event_id": "e-1",
                "step": "plan",
                "level": "INFO",
                "message": "planned",
                "ts_utc": "2026-06-07T10:00:01Z",
                "payload": {"tool": "rag.retrieve"},
            },
            {
                "event_id": "e-2",
                "step": "execute",
                "level": "ERROR",
                "message": "failed",
                "ts_utc": "2026-06-07T10:00:02Z",
                "payload": {},
            },
        ]

    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        if run_id != "run-1" or tenant_id != "tenant-a":
            raise KeyError("not found")
        return PersistedRun(metadata=self._metadata, events=list(self._events))

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> list[RunSummary]:
        return [
            RunSummary(
                run_id="run-1",
                tenant_id=tenant_id,
                user_id="u-1",
                session_id="sess-1",
                started_at_utc="2026-06-07T10:00:00Z",
                duration_ms=120,
                event_count=2,
            )
        ][:limit]


def test_harness_get_run_returns_metadata_and_events() -> None:
    ctx = ToolWiringContext(trace_reader=InMemoryTraceReader())
    out = harness_get_run(ctx, HarnessGetRunInput(run_id="run-1", tenant_id="tenant-a"))
    assert out.metadata.run_id == "run-1"
    assert out.metadata.error_type == RuntimeErrorCode.INTERNAL_ERROR.value
    assert out.event_count == 2


def test_harness_list_runs_returns_summaries() -> None:
    ctx = ToolWiringContext(trace_reader=InMemoryTraceReader())
    out = harness_list_runs(ctx, HarnessListRunsInput(tenant_id="tenant-a", limit=5))
    assert out.total == 1
    assert out.runs[0].event_count == 2


def test_harness_get_run_cost_returns_usage() -> None:
    ctx = ToolWiringContext(trace_reader=InMemoryTraceReader())
    out = harness_get_run_cost(ctx, HarnessGetRunCostInput(run_id="run-1", tenant_id="tenant-a"))
    assert out.duration_ms == 120
    assert out.llm_usage["input_tokens"] == 10


def test_harness_get_run_events_filters_by_level() -> None:
    ctx = ToolWiringContext(trace_reader=InMemoryTraceReader())
    out = harness_get_run_events(
        ctx,
        HarnessGetRunEventsInput(run_id="run-1", tenant_id="tenant-a", level="ERROR"),
    )
    assert out.total == 1
    assert out.events[0].step == "execute"


def test_harness_trace_reader_not_configured() -> None:
    with pytest.raises(RuntimeError, match="trace_reader_not_configured"):
        harness_get_run(ToolWiringContext(), HarnessGetRunInput(run_id="run-1", tenant_id="tenant-a"))
