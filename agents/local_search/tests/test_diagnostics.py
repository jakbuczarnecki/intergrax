# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from local_search.diagnostics import (
    SearchSummaryDiagnostic,
    SearchSummaryReason,
    parse_search_summary_reason,
    search_diagnostic_from_output,
)


@pytest.mark.unit
def test_search_summary_reason_frozen_values() -> None:
    assert {reason.value for reason in SearchSummaryReason} == {
        "query_missing",
        "tool_gateway_not_available",
        "retrieve_failed",
        "retrieve_complete",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            SearchSummaryReason.RETRIEVE_COMPLETE,
            SearchSummaryReason.RETRIEVE_COMPLETE,
        ),
        ("retrieve_complete", SearchSummaryReason.RETRIEVE_COMPLETE),
        (" retrieve_failed ", SearchSummaryReason.RETRIEVE_FAILED),
        ("query_missing", SearchSummaryReason.QUERY_MISSING),
        (
            "tool_gateway_not_available",
            SearchSummaryReason.TOOL_GATEWAY_NOT_AVAILABLE,
        ),
        ("unknown_reason", None),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
        (True, None),
    ],
)
def test_parse_search_summary_reason(
    value: object,
    expected: SearchSummaryReason | None,
) -> None:
    assert parse_search_summary_reason(value) is expected


@pytest.mark.unit
def test_search_summary_schema_id_and_redaction() -> None:
    payload = SearchSummaryDiagnostic(
        num_results=2,
        evidence_count=2,
        query_digest="abc123",
        query_text="secret query",
        source_refs=("docs/a.md",),
        used=True,
        reason=SearchSummaryReason.RETRIEVE_COMPLETE,
    )
    assert payload.schema_id() == "lkw.search_summary.v1"
    redacted = payload.redact()
    assert redacted.query_text is None
    assert redacted.query_digest == "abc123"
    assert redacted.used is True
    assert redacted.reason is SearchSummaryReason.RETRIEVE_COMPLETE
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
        reason=SearchSummaryReason.RETRIEVE_FAILED,
    )
    assert payload.reason is SearchSummaryReason.RETRIEVE_FAILED
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
    assert payload.reason is SearchSummaryReason.RETRIEVE_COMPLETE


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
    assert payload.reason is SearchSummaryReason.RETRIEVE_FAILED
    assert payload.to_dict()["reason"] == "retrieve_failed"
    assert payload.raw_tool_reason == "provider_cold"


@pytest.mark.unit
def test_search_diagnostic_from_output_rejects_unknown_reason() -> None:
    output: dict[str, object] = {
        "search_summary": {
            "used": True,
            "reason": "future_unknown_reason",
            "num_results": 0,
            "evidence": [],
        }
    }
    payload = search_diagnostic_from_output(output)
    assert payload.reason is None


@pytest.mark.unit
def test_search_summary_to_dict_is_json_compatible() -> None:
    payload = SearchSummaryDiagnostic(
        num_results=1,
        evidence_count=1,
        used=True,
        reason=SearchSummaryReason.RETRIEVE_COMPLETE,
    )
    serialized = payload.to_dict()
    encoded = json.dumps(serialized)
    assert '"reason": "retrieve_complete"' in encoded
    assert "SearchSummaryReason" not in encoded
    assert "RETRIEVE_COMPLETE" not in encoded
