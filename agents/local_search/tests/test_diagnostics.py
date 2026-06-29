# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_search.diagnostics import SearchSummaryDiagnostic, search_diagnostic_from_output


@pytest.mark.unit
def test_search_summary_schema_id_and_redaction() -> None:
    payload = SearchSummaryDiagnostic(
        num_results=2,
        evidence_count=2,
        query_digest="abc123",
        query_text="secret query",
        source_refs=("docs/a.md",),
    )
    assert payload.schema_id() == "lkw.search_summary.v1"
    redacted = payload.redact()
    assert redacted.query_text is None
    assert redacted.query_digest == "abc123"
    assert "query_text" not in redacted.to_dict()


@pytest.mark.unit
def test_search_diagnostic_from_output_maps_search_summary() -> None:
    output = {
        "search_summary": {
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
