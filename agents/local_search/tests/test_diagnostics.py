# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_search.diagnostics import (
    SearchSummaryDiagnostic,
    search_diagnostic_from_output,
)


@pytest.mark.unit
def test_search_summary_schema_id_and_redaction() -> None:
    payload = SearchSummaryDiagnostic(
        num_results=2,
        evidence_count=2,
        query_digest="abc123",
        query_text="secret query",
        source_refs=("docs/a.md",),
        used=True,
        reason="retrieve_complete",
    )
    assert payload.schema_id() == "lkw.search_summary.v1"
    redacted = payload.redact()
    assert redacted.query_text is None
    assert redacted.query_digest == "abc123"
    assert redacted.used is True
    assert redacted.reason == "retrieve_complete"
    assert "query_text" not in redacted.to_dict()
    serialized = redacted.to_dict()
    assert serialized["used"] is True
    assert serialized["reason"] == "retrieve_complete"


@pytest.mark.unit
def test_search_summary_to_dict_preserves_used_false() -> None:
    payload = SearchSummaryDiagnostic(
        num_results=0,
        evidence_count=0,
        used=False,
        reason="retrieve_failed",
    )
    serialized = payload.to_dict()
    assert serialized["used"] is False
    assert serialized["reason"] == "retrieve_failed"


@pytest.mark.unit
def test_search_diagnostic_from_output_maps_search_summary() -> None:
    output: dict[str, object] = {
        "search_summary": {
            "used": True,
            "reason": "retrieve_complete",
            "query": "find docs",
            "num_results": 1,
            "collection_id": "ws-1",
            "evidence": [{"source_path": "docs/a.md", "text": "chunk"}],
        }
    }
    payload = search_diagnostic_from_output(output)
    assert isinstance(payload, SearchSummaryDiagnostic)
    assert payload.num_results == 1
    assert payload.evidence_count == 1
    assert payload.workspace_id == "ws-1"
    assert payload.source_refs == ("docs/a.md",)
    assert payload.used is True
    assert payload.reason == "retrieve_complete"


@pytest.mark.unit
def test_search_diagnostic_from_output_maps_retrieve_failed() -> None:
    output: dict[str, object] = {
        "search_summary": {
            "used": False,
            "reason": "retrieve_failed",
            "num_results": 0,
            "raw_tool_reason": "provider_cold",
            "evidence": [],
        }
    }
    payload = search_diagnostic_from_output(output)
    assert payload.used is False
    assert payload.reason == "retrieve_failed"
    assert payload.raw_tool_reason == "provider_cold"
