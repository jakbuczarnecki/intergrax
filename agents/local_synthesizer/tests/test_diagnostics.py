# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_synthesizer.diagnostics import (
    SynthesizeSummaryDiagnostic,
    synthesize_diagnostic_from_output,
)


@pytest.mark.unit
def test_synthesize_summary_schema_id_and_fields() -> None:
    payload = SynthesizeSummaryDiagnostic(
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
    assert isinstance(payload, SynthesizeSummaryDiagnostic)
    assert payload.write_status == "write_complete"
    assert payload.shadow_write is True
    assert payload.source_evidence_count == 2
    assert payload.artifact_path == "draft.md"
