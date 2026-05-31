# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.parser_trace_span import append_parser_trace_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_append_parser_trace_event() -> None:
    store = InMemoryRunTraceStore()
    trace = {"parser_id": "docling.local", "attempts": [{"parser_id": "docling.local", "status": "success"}]}
    append_parser_trace_event(
        store,
        run_id="run-1",
        source="/tmp/doc.pdf",
        trace=trace,
        session_id="s1",
    )
    assert len(store._events_by_run["run-1"]) == 1
    event = store._events_by_run["run-1"][0]
    assert event.step == "document_parser"
    assert event.tags.get("parser_id") == "docling.local"
