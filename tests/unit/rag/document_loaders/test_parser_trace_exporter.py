# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import logging

import pytest

from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_export_parser_trace_logs(caplog: pytest.LogCaptureFixture) -> None:
    trace = {
        "parser_id": "pymupdf",
        "attempts": [{"parser_id": "pymupdf", "status": "success", "latency_ms": 12.0}],
    }
    with caplog.at_level(logging.INFO):
        export_parser_trace(source="/tmp/doc.pdf", trace=trace)
    assert any("document_parser_trace" in r.message for r in caplog.records)
