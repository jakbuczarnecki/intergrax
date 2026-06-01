# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.metrics.export import _summarize_trace_events
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

pytestmark = pytest.mark.gate


def test_summarize_uses_trace_component() -> None:
    events = [
        TraceEvent(
            event_id="e1",
            run_id="r1",
            seq=1,
            ts_utc="2026-06-01T00:00:00Z",
            level=TraceLevel.INFO,
            component=TraceComponent.ENGINE,
            step="core_llm",
            message="generate",
        ),
        TraceEvent(
            event_id="e2",
            run_id="r1",
            seq=2,
            ts_utc="2026-06-01T00:00:01Z",
            level=TraceLevel.INFO,
            component=TraceComponent.TOOLS,
            step="tools",
            message="invoke",
        ),
    ]
    summary = _summarize_trace_events(events)
    assert summary["llm_events"] >= 1
    assert summary["tool_events"] >= 1
