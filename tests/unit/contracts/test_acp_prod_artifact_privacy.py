# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.artifact_ref import ArtifactRef, artifact_ref_from_payload
from intergrax.contracts.privacy_redaction import redact_pii_text


@pytest.mark.unit
@pytest.mark.gate
def test_artifact_ref_from_payload() -> None:
    ref = artifact_ref_from_payload(
        {
            "artifact_id": "art-1",
            "type": "report",
            "uri": "file:///tmp/report.pdf",
            "tool_id": "doc.export",
            "sensitivity": "internal",
        },
        run_id="run-1",
        trace_id="trace-1",
        agent_id="legal",
        step_index=2,
    )
    assert isinstance(ref, ArtifactRef)
    assert ref.schema_version == "artifact_ref.v1"
    assert ref.provenance.created_by_agent_id == "legal"
    assert ref.provenance.created_by_tool_id == "doc.export"
    assert ref.step_index == 2


@pytest.mark.unit
@pytest.mark.gate
def test_redact_pii_text_masks_email_and_bearer() -> None:
    raw = "contact user@example.com token Bearer abc.def.ghi"
    redacted = redact_pii_text(raw)
    assert "user@example.com" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "[EMAIL]" in redacted
    assert "[REDACTED]" in redacted
