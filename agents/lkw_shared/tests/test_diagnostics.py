# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.authoring.diagnostic_serialization import merge_diagnostic_payloads
from lkw_shared.diagnostics import (
    LkwIndexSummaryDiagnostic,
    LkwSearchSummaryDiagnostic,
    LkwSynthesizeSummaryDiagnostic,
    index_diagnostic_from_output,
    search_diagnostic_from_output,
    synthesize_diagnostic_from_output,
)


@pytest.mark.unit
def test_lkw_index_summary_schema_id_and_required_fields() -> None:
    payload = LkwIndexSummaryDiagnostic(
        accepted_count=2,
        rejected_count=1,
        ingested_count=2,
        chunk_count=5,
        source_count=3,
    )
    assert payload.schema_id() == "lkw.index_summary.v1"
    data = payload.to_dict()
    assert data["accepted_count"] == 2
    assert data["ingested_count"] == 2
    assert data["chunk_count"] == 5


@pytest.mark.unit
def test_lkw_search_summary_schema_id_and_redaction() -> None:
    payload = LkwSearchSummaryDiagnostic(
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
def test_lkw_synthesize_summary_schema_id_and_fields() -> None:
    payload = LkwSynthesizeSummaryDiagnostic(
        write_status="write_complete",
        shadow_write=True,
        source_evidence_count=3,
        artifact_path="draft.md",
    )
    assert payload.schema_id() == "lkw.synthesize_summary.v1"
    data = payload.to_dict()
    assert data["write_status"] == "write_complete"
    assert data["shadow_write"] is True
    assert data["source_evidence_count"] == 3


@pytest.mark.unit
def test_index_diagnostic_from_output_maps_ingest_summary() -> None:
    output = {
        "ingest_summary": {
            "accepted_paths": ["a.txt"],
            "rejected_paths": [{"path": "b.txt", "reason": "source_not_found"}],
            "ingested": [{"status": "success", "num_chunks": 2}],
            "num_chunks": 2,
        }
    }
    payload = index_diagnostic_from_output(output)
    assert isinstance(payload, LkwIndexSummaryDiagnostic)
    assert payload.accepted_count == 1
    assert payload.rejected_count == 1
    assert payload.ingested_count == 1
    assert payload.chunk_count == 2
    assert payload.rejected_reasons == ("source_not_found",)


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
    assert isinstance(payload, LkwSearchSummaryDiagnostic)
    assert payload.num_results == 1
    assert payload.evidence_count == 1
    assert payload.workspace_id == "ws-1"
    assert payload.source_refs == ("docs/a.md",)


@pytest.mark.unit
def test_synthesize_diagnostic_from_output_maps_synthesize_summary() -> None:
    output = {
        "synthesize_summary": {
            "used": True,
            "reason": "write_complete",
            "shadow_workspace": True,
            "num_evidence_items": 2,
            "artifact_path": "draft.md",
            "output_name": "draft.md",
        }
    }
    payload = synthesize_diagnostic_from_output(output)
    assert isinstance(payload, LkwSynthesizeSummaryDiagnostic)
    assert payload.write_status == "write_complete"
    assert payload.shadow_write is True
    assert payload.source_evidence_count == 2
    assert payload.artifact_path == "draft.md"


@pytest.mark.unit
def test_merge_diagnostic_payloads_keys_by_schema_id() -> None:
    payload = LkwIndexSummaryDiagnostic(
        accepted_count=1,
        rejected_count=0,
        ingested_count=1,
        chunk_count=1,
        source_count=1,
    )
    merged = merge_diagnostic_payloads(None, [payload])
    assert "lkw.index_summary.v1" in merged
    assert merged["lkw.index_summary.v1"]["accepted_count"] == 1
