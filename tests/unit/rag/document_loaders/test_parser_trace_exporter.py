# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import logging

import pytest

from intergrax.rag.document_loaders.observability.parser_trace_contract import (
    DocumentParserTrace,
    ParserAttemptStatus,
    ParserTraceAttempt,
)
from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_export_parser_trace_logs(caplog: pytest.LogCaptureFixture) -> None:
    trace = DocumentParserTrace(
        parser_id="pymupdf",
        attempts=(
            ParserTraceAttempt(
                parser_id="pymupdf",
                status=ParserAttemptStatus.SUCCESS,
                latency_ms=12.0,
            ),
        ),
    )
    with caplog.at_level(logging.INFO):
        export_parser_trace(source="/tmp/doc.pdf", trace=trace)
    assert any("document_parser_trace" in r.message for r in caplog.records)


def test_export_parser_trace_deprecated_vendor_env_is_ignored(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_EXPORT_PARSER_TRACE", "1")
    monkeypatch.setenv("INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND", "sentry")
    trace = DocumentParserTrace(parser_id="pymupdf", attempts=())
    with caplog.at_level(logging.WARNING):
        export_parser_trace(source="/tmp/doc.pdf", trace=trace)
    assert any("deprecated parser trace vendor export env ignored" in r.message for r in caplog.records)
