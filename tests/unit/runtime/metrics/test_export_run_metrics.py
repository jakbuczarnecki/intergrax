# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.metrics.export import export_run_metrics, _summarize_trace_events
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolInvocationStartDiagV1

pytestmark = pytest.mark.gate


def test_summarize_trace_events_uses_diagnostic_schema_id() -> None:
    events = [
        TraceEvent(
            event_id="e1",
            run_id="r1",
            seq=1,
            ts_utc="2026-01-01T00:00:00Z",
            level=TraceLevel.INFO,
            component=TraceComponent.TOOLS,
            step="tool_invoke",
            message="tool ok",
            payload=ToolInvocationStartDiagV1(
                tool_id="rag.retrieve",
                step_id="s1",
                side_effects=False,
                input_payload={},
            ),
        ),
    ]
    summary = _summarize_trace_events(events)
    assert summary["tool_events"] >= 1


def test_export_run_metrics_from_persisted_run() -> None:
    persisted = PersistedRun(
        metadata=RunMetadata(
            run_id="run-1",
            session_id="sess",
            user_id="u",
            tenant_id="t",
            started_at_utc="2026-01-01T00:00:00Z",
            stats=RunStats(duration_ms=100, llm_usage={"total_tokens": 42}),
        ),
        events=[],
    )
    exported = export_run_metrics(persisted, agent_id="agent-1")
    assert exported.run_id == "run-1"
    assert exported.total_tokens == 42
