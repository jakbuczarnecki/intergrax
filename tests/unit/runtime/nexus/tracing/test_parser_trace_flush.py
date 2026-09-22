# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from intergrax.contracts.persisted_run_trace import PersistedTraceEvent
from intergrax.contracts.tracing import TraceComponent, TraceEvent, TraceLevel
from intergrax.rag.document_loaders.observability.parser_trace_contract import (
    DocumentParserTrace,
    ParserAttemptStatus,
    ParserTraceAttempt,
)
from intergrax.runtime.nexus.tracing.parser_trace_flush import export_parser_traces_from_events
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent

pytestmark = pytest.mark.gate


def _sample_trace() -> DocumentParserTrace:
    return DocumentParserTrace(
        parser_id="parse",
        attempts=(ParserTraceAttempt(parser_id="parse", status=ParserAttemptStatus.SUCCESS),),
    )


def test_export_parser_traces_reads_tags_from_trace_event() -> None:
    trace = _sample_trace()
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id="run-1",
        seq=0,
        ts_utc="2026-01-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.RAG,
        step="document_parser",
        message="document_parser:parse",
        tags={
            "integration_parser_trace": trace.to_logging_extra_value(),
            "source": "unit_test",
        },
    )
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events([event])
        export_mock.assert_called_once_with(source="unit_test", trace=trace)


def test_export_parser_traces_reads_tags_from_serialized_event() -> None:
    trace_payload = _sample_trace().to_logging_extra_value()
    event = SerializedTraceEvent(
        event_id="e1",
        run_id="run-1",
        seq=0,
        ts_utc="2026-01-01T00:00:00Z",
        level="info",
        component="rag",
        step="document_parser",
        message="document_parser:parse",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"integration_parser_trace": trace_payload, "source": "serialized"},
        artifact_refs=[],
    )
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events([event])
        export_mock.assert_called_once()
        assert export_mock.call_args.kwargs["source"] == "serialized"


def test_export_parser_traces_reads_tags_from_persisted_event() -> None:
    trace_payload = _sample_trace().to_logging_extra_value()
    event = PersistedTraceEvent(
        event_id="e1",
        run_id="run-1",
        seq=0,
        ts_utc="2026-01-01T00:00:00Z",
        level="info",
        component="rag",
        step="document_parser",
        message="document_parser:parse",
        tags={"integration_parser_trace": trace_payload, "source": "persisted"},
    )
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events([event])
        export_mock.assert_called_once_with(source="persisted", trace=_sample_trace())


def test_export_parser_traces_ignores_events_without_trace_tag() -> None:
    event = PersistedTraceEvent(
        event_id="e1",
        run_id="run-1",
        seq=0,
        ts_utc="2026-01-01T00:00:00Z",
        level="info",
        component="rag",
        step="other",
        message="other",
        tags={"other": True},
    )
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events([event])
        export_mock.assert_not_called()
