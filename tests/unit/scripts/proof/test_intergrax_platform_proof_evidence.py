# Â© Artur Czarnecki. All rights reserved.

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
    ReportSafeEvidenceBackedClaim,
    ReportSafeEvidenceChallenge,
    ReportSafeEvidenceClaimSet,
    ReportSafeField,
    ReportSafePayload,
    ReportSafeTextSourceKind,
    ReportSafeVisibility,
    ReproductionEvidence,
    ToolInvocationEvidence,
    explicit_runtime_report_safe_text,
    project_evidence_backed_claim,
    project_evidence_challenge,
    project_evidence_claim_set,
    proof_authored_report_safe_text,
    redacted_report_safe_text,
    sanitized_runtime_report_safe_text,
    sanitize_untrusted_report_text,
    execution_status_from_proof_status,
)
from intergrax.contracts.evidence_claims import (
    ChallengeDefectFamily,
    ChallengeResolution,
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceChallenge,
    EvidenceClaimSet,
    validate_claim_kind,
    validate_defect_code,
    validate_evidence_challenge_id,
    validate_evidence_claim_id,
    validate_evidence_reference_id,
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
                domains_exercised=("test_domain",),
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


def test_proof_identity_domains_exercised_canonical_lexicographic_order() -> None:
    identity_a = ProofIdentityEvidence(
        proof_id="P",
        title="t",
        domains_exercised=["TOOLS", "EXECUTION", "OBSERVABILITY"],
        proof_version="v1",
        source_revision="sha",
        execution_profile=ProofProfile.QUICK,
    )
    identity_b = ProofIdentityEvidence(
        proof_id="P",
        title="t",
        domains_exercised=["OBSERVABILITY", "TOOLS", "EXECUTION"],
        proof_version="v1",
        source_revision="sha",
        execution_profile=ProofProfile.QUICK,
    )
    expected = ("EXECUTION", "OBSERVABILITY", "TOOLS")
    assert identity_a.domains_exercised == expected
    assert identity_b.domains_exercised == expected


def test_proof_identity_domains_exercised_trimming_before_canonical_order() -> None:
    identity = ProofIdentityEvidence(
        proof_id="P",
        title="t",
        domains_exercised=["  TOOLS  ", "EXECUTION", " OBSERVABILITY "],
        proof_version="v1",
        source_revision="sha",
        execution_profile=ProofProfile.QUICK,
    )
    assert identity.domains_exercised == ("EXECUTION", "OBSERVABILITY", "TOOLS")


def _claim_id(suffix: str = "0123456789abcdef0123456789abcdef") -> str:
    return f"eclaim_{suffix}"


def _challenge_id(suffix: str = "fedcba9876543210fedcba9876543210") -> str:
    return f"echlg_{suffix}"


def _manufacturing_claim_set() -> EvidenceClaimSet:
    claim_id = validate_evidence_claim_id(_claim_id())
    return EvidenceClaimSet(
        claims=(
            EvidenceBackedClaim(
                claim_id=claim_id,
                statement="Equipment degradation caused the incident.",
                claim_kind=validate_claim_kind("manufacturing.root_cause"),
                supporting_evidence_ids=[validate_evidence_reference_id("e1")],
                contradicting_evidence_ids=[validate_evidence_reference_id("e2")],
                resolution=ClaimResolution.REJECTED,
            ),
        ),
        challenges=(
            EvidenceChallenge(
                challenge_id=validate_evidence_challenge_id(_challenge_id()),
                claim_id=claim_id,
                defect_family=ChallengeDefectFamily.UNSUPPORTED_INFERENCE,
                defect_code=validate_defect_code("manufacturing.correlation_only"),
                evidence_ids=[validate_evidence_reference_id("e3")],
                description="Correlation is insufficient to establish causation.",
                resolution=ChallengeResolution.SATISFIED,
            ),
        ),
    )


def _tprm_claim_set() -> EvidenceClaimSet:
    claim_id = validate_evidence_claim_id(
        "eclaim_abcdef0123456789abcdef0123456789"
    )
    return EvidenceClaimSet(
        claims=(
            EvidenceBackedClaim(
                claim_id=claim_id,
                statement="Vendor SOC 2 report covers required controls.",
                claim_kind=validate_claim_kind("tprm.vendor_assurance"),
                supporting_evidence_ids=[validate_evidence_reference_id("vendor-report")],
                resolution=ClaimResolution.SUPPORTED,
            ),
        ),
    )


def test_projection_preserves_structural_fields_manufacturing() -> None:
    canonical = _manufacturing_claim_set()
    projected = project_evidence_claim_set(
        canonical,
        text_source=ReportSafeTextSourceKind.PROOF_AUTHORED,
    )
    source_claim = canonical.claims[0]
    target_claim = projected.claims[0]
    source_challenge = canonical.challenges[0]
    target_challenge = projected.challenges[0]
    assert target_claim.claim_id == source_claim.claim_id
    assert target_claim.claim_kind == source_claim.claim_kind
    assert target_claim.supporting_evidence_ids == source_claim.supporting_evidence_ids
    assert target_claim.contradicting_evidence_ids == source_claim.contradicting_evidence_ids
    assert target_claim.resolution == source_claim.resolution
    assert target_claim.supersedes_claim_id == source_claim.supersedes_claim_id
    assert target_challenge.challenge_id == source_challenge.challenge_id
    assert target_challenge.claim_id == source_challenge.claim_id
    assert target_challenge.defect_family == source_challenge.defect_family
    assert target_challenge.defect_code == source_challenge.defect_code
    assert target_challenge.evidence_ids == source_challenge.evidence_ids
    assert target_challenge.resolution == source_challenge.resolution


def test_projection_preserves_structural_fields_tprm() -> None:
    canonical = _tprm_claim_set()
    projected = project_evidence_claim_set(
        canonical,
        text_source=ReportSafeTextSourceKind.RUNTIME_SANITIZED,
    )
    assert projected.claims[0].claim_kind == canonical.claims[0].claim_kind
    assert projected.claims[0].supporting_evidence_ids == canonical.claims[0].supporting_evidence_ids


def test_runtime_claim_statement_cannot_bypass_report_safe_text() -> None:
    claim = EvidenceBackedClaim(
        claim_id=validate_evidence_claim_id(_claim_id()),
        statement="Authorization: Bearer fake-secret-value",
        claim_kind=validate_claim_kind("generic.claim"),
    )
    with pytest.raises(ValueError, match="raw runtime string"):
        ReportSafeEvidenceBackedClaim(
            claim_id=claim.claim_id,
            statement=claim.statement,
            claim_kind=claim.claim_kind,
        )


def test_runtime_claim_statement_redacts_known_secret_patterns() -> None:
    claim = EvidenceBackedClaim(
        claim_id=validate_evidence_claim_id(_claim_id()),
        statement="Authorization: Bearer fake-secret-value",
        claim_kind=validate_claim_kind("generic.claim"),
    )
    projected = project_evidence_backed_claim(
        claim,
        statement_source=ReportSafeTextSourceKind.RUNTIME_SANITIZED,
    )
    assert "fake-secret-value" not in projected.statement.text
    assert projected.statement.redaction_applied is True


def test_challenge_description_obey_runtime_report_safe_rule() -> None:
    challenge = EvidenceChallenge(
        challenge_id=validate_evidence_challenge_id(_challenge_id()),
        claim_id=validate_evidence_claim_id(_claim_id()),
        defect_family=ChallengeDefectFamily.MISSING_EVIDENCE,
        description="OPENAI_API_KEY=fake-secret",
    )
    projected = project_evidence_challenge(
        challenge,
        description_source=ReportSafeTextSourceKind.RUNTIME_SANITIZED,
    )
    assert "fake-secret" not in projected.description.text
    assert projected.description.redaction_applied is True


def test_proof_authored_claim_text_unchanged() -> None:
    claim = EvidenceBackedClaim(
        claim_id=validate_evidence_claim_id(_claim_id()),
        statement="Static proof-authored claim text.",
        claim_kind=validate_claim_kind("generic.claim"),
    )
    projected = project_evidence_backed_claim(
        claim,
        statement_source=ReportSafeTextSourceKind.PROOF_AUTHORED,
    )
    assert projected.statement.text == claim.statement
    assert projected.statement.redaction_applied is False


def test_empty_evidence_claims_default_for_existing_proofs() -> None:
    evidence = _minimal_platform_evidence()
    assert evidence.evidence_claims.claims == ()
    assert evidence.evidence_claims.challenges == ()


def test_evidence_claims_reject_dangling_support_reference() -> None:
    claim_set = project_evidence_claim_set(
        _manufacturing_claim_set(),
        text_source=ReportSafeTextSourceKind.PROOF_AUTHORED,
    )
    with pytest.raises(ValueError, match="claim_support_evidence_missing"):
        _minimal_platform_evidence(evidence_claims=claim_set)


def test_evidence_claims_json_round_trip() -> None:
    claim_set = project_evidence_claim_set(
        _manufacturing_claim_set(),
        text_source=ReportSafeTextSourceKind.PROOF_AUTHORED,
    )
    evidence = _minimal_platform_evidence(
        evidence_claims=claim_set,
        evidence_graph=_evidence_graph_for_claim_set(claim_set),
    )
    payload = evidence.model_dump(mode="json")
    restored = PlatformProofEvidence.model_validate(payload)
    assert restored.evidence_claims == evidence.evidence_claims


def _evidence_graph_for_claim_set(
    claim_set: ReportSafeEvidenceClaimSet,
) -> EvidenceGraphEvidence:
    evidence_ids: set[str] = set()
    for claim in claim_set.claims:
        evidence_ids.update(str(item) for item in claim.supporting_evidence_ids)
        evidence_ids.update(str(item) for item in claim.contradicting_evidence_ids)
    for challenge in claim_set.challenges:
        evidence_ids.update(str(item) for item in challenge.evidence_ids)
    return EvidenceGraphEvidence(
        nodes=tuple(
            EvidenceNode(
                evidence_id=evidence_id,
                kind=EvidenceNodeKind.OTHER,
                label=evidence_id,
            )
            for evidence_id in sorted(evidence_ids)
        )
    )


def _minimal_platform_evidence(
    *,
    evidence_claims: ReportSafeEvidenceClaimSet | None = None,
    evidence_graph: EvidenceGraphEvidence | None = None,
) -> PlatformProofEvidence:
    return PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id="P",
            title="t",
            domains_exercised=("test_domain",),
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
            claim="proof-level claim",
            user_relevance="u",
            success_criteria=("s",),
            falsification_criteria=("f",),
            excluded_claims=("e",),
        ),
        architecture=_minimal_architecture(),
        participants=_minimal_architecture().participants,
        environment=_minimal_environment(),
        evidence_graph=evidence_graph or EvidenceGraphEvidence(),
        evidence_claims=evidence_claims or ReportSafeEvidenceClaimSet(),
        reproduction=_minimal_reproduction(),
        provenance=_minimal_provenance(),
    )
