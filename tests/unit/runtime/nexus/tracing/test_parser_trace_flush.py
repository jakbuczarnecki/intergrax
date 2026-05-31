# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.parser_trace_span import append_parser_trace_event
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_finalize_run_exports_parser_trace_spans() -> None:
    store = InMemoryRunTraceStore()
    append_parser_trace_event(
        store,
        run_id="run-1",
        source="test.pdf",
        trace={"parser_id": "docling", "attempts": []},
    )
    metadata = RunMetadata(
        run_id="run-1",
        session_id="s1",
        user_id="u1",
        tenant_id="t1",
        started_at_utc="2026-05-30T00:00:00Z",
        stats=RunStats(duration_ms=1, llm_usage={}),
    )
    with patch("intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace") as export_mock:
        store.finalize_run("run-1", metadata)
    export_mock.assert_called_once()
