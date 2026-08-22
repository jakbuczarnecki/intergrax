# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEvidence,
    EnvironmentEvidence,
    EvidenceEdge,
    EvidenceGraphEvidence,
    EvidenceNode,
    EvidenceNodeKind,
    EvidenceRelationship,
    ExecutionMetadataEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofIdentityEvidence,
    ProvenanceEvidence,
    ReportSafeField,
    ReportSafePayload,
    ReportSafeTextSourceKind,
    ReportSafeVisibility,
    ReproductionEvidence,
    ToolInvocationEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    redacted_report_safe_text,
    sanitized_runtime_report_safe_text,
    sanitize_untrusted_report_text,
    execution_status_from_proof_status,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile, ProofStatus, SuiteReceipt

pytestmark = pytest.mark.unit


def test_report_safe_payload_rejects_secret_field() -> None:
    with pytest.raises(ValueError, match="secret-bearing field"):
        ReportSafePayload(
            fields=(
                ReportSafeField(
                    name="api_key",
                    visibility=ReportSafeVisibility.REPORT_SAFE,
                    value=proof_authored_report_safe_text("super-secret"),
                ),
            )
        )


def test_report_safe_field_rejects_raw_runtime_string() -> None:
    with pytest.raises(ValueError, match="raw runtime string"):
        ReportSafeField(
            name="output",
            visibility=ReportSafeVisibility.REPORT_SAFE,
            value="Authorization: Bearer fake-secret-value",
        )


def test_report_safe_text_serializes() -> None:
    safe = explicit_runtime_report_safe_text("North express long_haul rate 0.68")
    payload = safe.model_dump(mode="json")
    assert payload["text"] == "North express long_haul rate 0.68"
    assert payload["visibility"] == "REPORT_SAFE"
    assert payload["source_kind"] == "RUNTIME_EXPLICIT"


def test_sanitize_bearer_header() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "Authorization: Bearer fake-secret-value"
    )
    assert redaction_applied is True
    assert "fake-secret-value" not in sanitized
    assert "Bearer [REDACTED]" in sanitized


def test_sanitize_env_assignment() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "missing config OPENAI_API_KEY=fake-secret"
    )
    assert redaction_applied is True
    assert "fake-secret" not in sanitized
    assert "OPENAI_API_KEY=[REDACTED]" in sanitized


def test_sanitize_credential_url() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "connect postgresql://proof_user:proof_pass@localhost:5432/proof_db"
    )
    assert redaction_applied is True
    assert "proof_pass" not in sanitized
    assert "[REDACTED]:[REDACTED]@" in sanitized


def test_ordinary_sql_unchanged() -> None:
    sql = (
        "SELECT service_type, route_type, AVG(delayed::int) "
        "FROM proof.parcel_events WHERE region='North' "
        "GROUP BY service_type, route_type"
    )
    safe = explicit_runtime_report_safe_text(sql)
    assert safe.text == sql
    assert safe.redaction_applied is False


def test_ordinary_sql_output_preview_unchanged() -> None:
    preview = "North express long_haul rate 0.68"
    safe = explicit_runtime_report_safe_text(preview)
    assert safe.text == preview
    assert safe.redaction_applied is False


def test_provider_error_without_secret_remains_readable() -> None:
    message = "400 Invalid tools[0].name"
    safe = sanitized_runtime_report_safe_text(message)
    assert safe.text == message
    assert safe.redaction_applied is False


def test_provider_error_with_fake_token_redacted_in_serialization() -> None:
    message = "Authorization: Bearer fake-secret-value"
    safe = sanitized_runtime_report_safe_text(message)
    payload = safe.model_dump(mode="json")
    assert "fake-secret-value" not in payload["text"]
    assert payload["redaction_applied"] is True


def test_final_model_answer_explicitly_report_safe() -> None:
    safe = explicit_runtime_report_safe_text("final answer")
    assert safe.source_kind is ReportSafeTextSourceKind.RUNTIME_EXPLICIT


def test_redacted_text_does_not_expose_original_value() -> None:
    safe = redacted_report_safe_text()
    assert safe.text == "[REDACTED]"
    assert safe.redaction_applied is True


def test_tool_invocation_rejects_raw_runtime_summary() -> None:
    with pytest.raises(ValueError, match="raw runtime string"):
        ToolInvocationEvidence(
            tool_id="sql-query",
            success=True,
            output_summary="Authorization: Bearer fake-secret-value",
        )


def test_evidence_graph_rejects_dangling_references() -> None:
    with pytest.raises(ValueError, match="dangling from_evidence_id"):
        PlatformProofEvidence(
            proof_identity=ProofIdentityEvidence(
                proof_id="P",
                title="t",
                domain="test_domain",
                proof_version="v1",
                source_revision="sha",
                execution_profile=ProofProfile.QUICK,
            ),
            execution=ExecutionMetadataEvidence(
                status=ProofEvidenceExecutionStatus.PASS,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                finished_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                platform="linux",
            ),
            claim=ProofClaimEvidence(
                claim="c",
                user_relevance="u",
                success_criteria=("s",),
                falsification_criteria=("f",),
                excluded_claims=("e",),
            ),
            architecture=_minimal_architecture(),
            participants=_minimal_architecture().participants,
            environment=_minimal_environment(),
            evidence_graph=EvidenceGraphEvidence(
                nodes=(
                    EvidenceNode(
                        evidence_id="evidence-a",
                        kind=EvidenceNodeKind.OTHER,
                        label="a",
                    ),
                ),
                edges=(
                    EvidenceEdge(
                        from_evidence_id="missing",
                        to_evidence_id="evidence-a",
                        relationship=EvidenceRelationship.EVIDENCE_BASIS,
                    ),
                ),
            ),
            reproduction=_minimal_reproduction(),
            provenance=_minimal_provenance(),
        )


def test_suite_receipt_contract_unchanged() -> None:
    assert SuiteReceipt.model_fields["schema_version"].default == "intergrax.proof_suite_receipt.v1"


def test_no_html_renderer_module_present() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    assert not (repo_root / "scripts" / "proof" / "platform_proof_report_renderer.py").exists()


def test_execution_status_mapping_from_proof_status() -> None:
    assert execution_status_from_proof_status(ProofStatus.PASS) is ProofEvidenceExecutionStatus.PASS
    assert (
        execution_status_from_proof_status(ProofStatus.BLOCKED_ENVIRONMENT)
        is ProofEvidenceExecutionStatus.BLOCKED
    )


def _minimal_architecture() -> ArchitectureEvidence:
    from scripts.proof.intergrax_platform_proof_evidence import (
        ParticipantClass,
        ParticipantEvidence,
    )

    participant = ParticipantEvidence(
        participant_id="p1",
        name="n",
        implementation="i",
        version_or_model="v",
        role="r",
        participant_class=ParticipantClass.PLATFORM,
    )
    return ArchitectureEvidence(participants=(participant,))


def _minimal_environment() -> EnvironmentEvidence:
    return EnvironmentEvidence()


def _minimal_reproduction() -> ReproductionEvidence:
    return ReproductionEvidence(
        source_revision="sha",
        command="uv run python run_proof.py",
    )


def _minimal_provenance() -> ProvenanceEvidence:
    return ProvenanceEvidence(
        proof_id="P",
        source_revision="sha",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution_id="exec-1",
        artifact_identity="artifact-1",
    )
