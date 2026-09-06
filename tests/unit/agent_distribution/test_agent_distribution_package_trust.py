# © Artur Czarnecki. All rights reserved.

"""AP-5 AgentPackageTrust coordinator tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.core.qualification import QualificationStatus
from pydantic import ValidationError

from intergrax.agent_distribution.catalog import (
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.identity import (
    AgentPackageCandidate,
    AgentPackageIdentity,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.core.qualification import QualificationEvidence
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentInstallationTrustRecord,
    AgentPackageQualificationResult,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from testing_support.agent_package_attestation import (
    build_test_attestation_trust_coordinator,
    verified_signature_qualification_evidence,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_FIXED_AT = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
_QUALIFIED_AT = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)

_PACKAGE = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
)
_PACKAGE_B = _PACKAGE.model_copy(
    update={"package_version": "2.0.0", "package_digest": _DIGEST_B}
)
_PUBLISHER = AgentPublisherIdentity(publisher_id="publisher:acme", display_name="ACME")
_BUILTIN_SOURCE = CatalogSourceIdentity(
    catalog_source_id="builtin",
    provider_kind=CatalogProviderKind.BUILTIN,
)
_LOCAL_SOURCE = CatalogSourceIdentity(
    catalog_source_id="local-dev",
    provider_kind=CatalogProviderKind.LOCAL_DEVELOPER,
)


def _production_policy(**overrides: object) -> AgentPackageTrustPolicy:
    base = {
        "posture": AgentPackageTrustPosture.PRODUCTION,
        "trust_profile_ref": "profile:production",
        "permitted_provider_kinds": frozenset(
            {CatalogProviderKind.BUILTIN, CatalogProviderKind.OFFICIAL_CATALOG}
        ),
        "permitted_delivery_sources": frozenset(
            {AgentDeliverySource.BUILTIN, AgentDeliverySource.MARKETPLACE}
        ),
        "required_evidence_kinds": frozenset(
            {
                AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                AgentQualificationEvidenceKind.REVOCATION_CHECK,
            }
        ),
    }
    base.update(overrides)
    return AgentPackageTrustPolicy(**base)


def _development_policy(**overrides: object) -> AgentPackageTrustPolicy:
    base = {
        "posture": AgentPackageTrustPosture.DEVELOPMENT,
        "trust_profile_ref": "profile:development",
        "permitted_provider_kinds": frozenset({CatalogProviderKind.LOCAL_DEVELOPER}),
        "permitted_delivery_sources": frozenset({AgentDeliverySource.LOCAL_DEVELOPER}),
        "required_evidence_kinds": frozenset(
            {AgentQualificationEvidenceKind.CONTRACT_VALIDATION}
        ),
        "required_qualification_status": QualificationStatus.QUALIFIED,
    }
    base.update(overrides)
    return AgentPackageTrustPolicy(**base)


def _qualification(
    *,
    status: QualificationStatus = QualificationStatus.PRODUCTION_QUALIFIED,
    publisher: AgentPublisherIdentity = _PUBLISHER,
    delivery_source: AgentDeliverySource = AgentDeliverySource.BUILTIN,
    qualified_at: datetime = _QUALIFIED_AT,
) -> AgentPackageQualificationResult:
    return AgentPackageQualificationResult(
        publisher=publisher,
        status=status,
        evidence=(
            verified_signature_qualification_evidence(
                package_identity=_PACKAGE,
                publisher_id=publisher.publisher_id,
            ),
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="qualified by test evidence",
        delivery_source=delivery_source,
        qualified_at=qualified_at,
    )


def _evaluate(**overrides: object):
    coordinator = build_test_attestation_trust_coordinator()
    params = {
        "package_identity": _PACKAGE,
        "catalog_source": _BUILTIN_SOURCE,
        "delivery_source": AgentDeliverySource.BUILTIN,
        "publisher": _PUBLISHER,
        "policy": _production_policy(),
        "qualification": _qualification(),
        "evidence_package_digest": _DIGEST_A,
        "evidence_id": "evidence:pkg-a",
        "evaluated_at": _FIXED_AT,
    }
    params.update(overrides)
    return coordinator.evaluate(**params)


def test_trust_happy_path_produces_installable_record() -> None:
    decision = _evaluate()
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW
    assert decision.reason_code is AgentPackageTrustReasonCode.QUALIFIED
    assert decision.installable is True
    assert decision.trust_record is not None
    assert decision.trust_record.package_digest == _DIGEST_A
    assert (
        decision.trust_record.qualification_status
        is QualificationStatus.PRODUCTION_QUALIFIED
    )
    assert decision.trust_record.publisher_identity_ref == "publisher:acme"
    assert len(decision.trust_evidence_refs) == 2


def test_trust_missing_evidence_digest_fails_closed() -> None:
    decision = _evaluate(evidence_package_digest=None)
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert (
        decision.reason_code
        is AgentPackageTrustReasonCode.MISSING_PACKAGE_DIGEST_EVIDENCE
    )


def test_trust_evidence_digest_match_may_allow() -> None:
    decision = _evaluate(evidence_package_digest=_DIGEST_A)
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW
    assert decision.reason_code is AgentPackageTrustReasonCode.QUALIFIED


def test_trust_evidence_digest_mismatch_fails_closed() -> None:
    decision = _evaluate(evidence_package_digest=_DIGEST_B)
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH


def test_trust_evidence_for_another_package_fails_closed() -> None:
    decision = _evaluate(evidence_package_digest=_DIGEST_B)
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH


def test_same_package_version_different_digest_cannot_reuse_qualification() -> None:
    """Digest replay: same id/version with digest B must not reuse evidence for digest A."""
    same_version_other_digest = _PACKAGE.model_copy(
        update={"package_digest": _DIGEST_B}
    )
    decision = build_test_attestation_trust_coordinator().evaluate(
        package_identity=same_version_other_digest,
        catalog_source=_BUILTIN_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=_production_policy(),
        qualification=_qualification(),
        evidence_package_digest=_DIGEST_A,
        evidence_id="evidence:pkg-a",
        evaluated_at=_FIXED_AT,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH


def test_trust_version_label_without_digest_cannot_authorize() -> None:
    coordinator = AgentPackageTrustCoordinator()
    candidate = AgentPackageCandidate(
        distribution_package_id="intergrax-local-search-agent",
        package_version="1.0.0",
    )
    decision = coordinator.evaluate_candidate(
        package_candidate=candidate,
        catalog_source=_BUILTIN_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=_production_policy(),
        qualification=_qualification(),
        evaluated_at=_FIXED_AT,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert (
        decision.reason_code is AgentPackageTrustReasonCode.VERSION_LABEL_WITHOUT_DIGEST
    )


def test_trust_publisher_mismatch_fails_closed() -> None:
    other = AgentPublisherIdentity(publisher_id="publisher:other")
    decision = _evaluate(
        publisher=_PUBLISHER,
        qualification=_qualification(publisher=other),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.PUBLISHER_MISMATCH


def test_trust_denied_publisher_fails_closed() -> None:
    policy = _production_policy(denied_publisher_ids=frozenset({"publisher:acme"}))
    decision = _evaluate(policy=policy)
    assert decision.reason_code is AgentPackageTrustReasonCode.PUBLISHER_DENIED


def test_trust_revoked_publisher_fails_closed() -> None:
    revocation = AgentPackageTrustRevocationState(
        revoked_publisher_ids=frozenset({"publisher:acme"})
    )
    decision = _evaluate(revocation_state=revocation)
    assert decision.reason_code is AgentPackageTrustReasonCode.PUBLISHER_REVOKED


def test_trust_disallowed_catalog_source_fails_closed() -> None:
    decision = _evaluate(catalog_source=_LOCAL_SOURCE)
    assert decision.reason_code is AgentPackageTrustReasonCode.SOURCE_NOT_PERMITTED


def test_trust_development_source_accepted_only_when_policy_permits() -> None:
    coordinator = AgentPackageTrustCoordinator()
    dev_qualification = AgentPackageQualificationResult(
        publisher=_PUBLISHER,
        status=QualificationStatus.QUALIFIED,
        evidence=(
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.CONTRACT_VALIDATION,
                code="contract_ok",
            ),
        ),
        reason="development qualification",
        delivery_source=AgentDeliverySource.LOCAL_DEVELOPER,
        qualified_at=_QUALIFIED_AT,
    )
    allowed = coordinator.evaluate(
        package_identity=_PACKAGE,
        catalog_source=_LOCAL_SOURCE,
        delivery_source=AgentDeliverySource.LOCAL_DEVELOPER,
        publisher=_PUBLISHER,
        policy=_development_policy(),
        qualification=dev_qualification,
        evidence_package_digest=_DIGEST_A,
        evidence_id="evidence:dev",
        evaluated_at=_FIXED_AT,
    )
    assert allowed.outcome is AgentPackageTrustOutcome.ALLOW

    denied = coordinator.evaluate(
        package_identity=_PACKAGE,
        catalog_source=_LOCAL_SOURCE,
        delivery_source=AgentDeliverySource.LOCAL_DEVELOPER,
        publisher=_PUBLISHER,
        policy=_production_policy(),
        qualification=dev_qualification,
        evidence_package_digest=_DIGEST_A,
        evaluated_at=_FIXED_AT,
    )
    assert denied.reason_code is AgentPackageTrustReasonCode.SOURCE_NOT_PERMITTED


def test_trust_missing_required_qualification_fails_closed() -> None:
    decision = _evaluate(qualification=None)
    assert decision.reason_code is AgentPackageTrustReasonCode.MISSING_REQUIRED_EVIDENCE


def test_trust_insufficient_qualification_status_fails_closed() -> None:
    decision = _evaluate(
        qualification=_qualification(status=QualificationStatus.QUALIFIED),
    )
    assert (
        decision.reason_code
        is AgentPackageTrustReasonCode.INSUFFICIENT_QUALIFICATION_STATUS
    )


def test_trust_revoked_evidence_fails_closed() -> None:
    revocation = AgentPackageTrustRevocationState(
        revoked_evidence_ids=frozenset({"evidence:pkg-a"})
    )
    decision = _evaluate(revocation_state=revocation)
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_REVOKED


def test_trust_evidence_delivery_source_mismatch_fails_closed() -> None:
    decision = _evaluate(
        qualification=_qualification(delivery_source=AgentDeliverySource.MARKETPLACE),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_PACKAGE_MISMATCH


def test_trust_revoked_package_digest_fails_closed() -> None:
    revocation = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset({_DIGEST_A})
    )
    decision = _evaluate(revocation_state=revocation)
    assert decision.reason_code is AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED


def test_trust_decision_is_deterministic() -> None:
    first = _evaluate()
    second = _evaluate()
    assert first.to_audit_dict() == second.to_audit_dict()


def test_installation_service_rejects_unacceptable_trust_record() -> None:
    state = AgentDistributionStoreState()
    service = InstallationService(InMemoryAgentInstallationStore(state))
    service.create_candidate_installation(
        installation_id="inst-1",
        installation_slot_id="slot-1",
        environment_id="env-1",
        package_identity=_PACKAGE,
    )
    with pytest.raises(AgentPackageTrustError):
        service.mark_verified(
            "inst-1",
            artifact_store_ref="store://artifacts/1",
            trust_record=AgentInstallationTrustRecord(
                qualification_status=QualificationStatus.NOT_QUALIFIED,
                package_digest=_DIGEST_A,
                publisher_identity_ref="publisher:acme",
                source_provider_id="builtin",
                trust_evidence_refs=(
                    AgentTrustEvidenceRef(
                        evidence_id="evidence:bad",
                        kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    ),
                ),
            ),
        )


def test_malformed_trust_record_digest_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentInstallationTrustRecord(
            qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
            package_digest="not-a-digest",
            publisher_identity_ref="publisher:acme",
            source_provider_id="builtin",
            trust_evidence_refs=(
                AgentTrustEvidenceRef(
                    evidence_id="evidence:bad",
                    kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                ),
            ),
        )


def test_trust_record_matching_digest_verifies_installation() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None

    state = AgentDistributionStoreState()
    service = InstallationService(InMemoryAgentInstallationStore(state))
    service.create_candidate_installation(
        installation_id="inst-a",
        installation_slot_id="slot-1",
        environment_id="env-1",
        package_identity=_PACKAGE,
    )
    verified = service.mark_verified(
        "inst-a",
        artifact_store_ref="store://artifacts/a",
        trust_record=decision.trust_record,
    )
    assert verified.value.installation_state is InstallationState.VERIFIED


def test_trust_record_digest_mismatch_cannot_verify_installation() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None

    state = AgentDistributionStoreState()
    service = InstallationService(InMemoryAgentInstallationStore(state))
    service.create_candidate_installation(
        installation_id="inst-b",
        installation_slot_id="slot-1",
        environment_id="env-1",
        package_identity=_PACKAGE_B,
    )
    with pytest.raises(AgentPackageTrustError):
        service.mark_verified(
            "inst-b",
            artifact_store_ref="store://artifacts/b",
            trust_record=decision.trust_record,
        )
    record = state.installations["inst-b"]
    assert record.installation_state is InstallationState.CANDIDATE


def test_installation_service_accepts_coordinator_trust_record() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None

    state = AgentDistributionStoreState()
    service = InstallationService(InMemoryAgentInstallationStore(state))
    service.create_candidate_installation(
        installation_id="inst-1",
        installation_slot_id="slot-1",
        environment_id="env-1",
        package_identity=_PACKAGE,
    )
    verified = service.mark_verified(
        "inst-1",
        artifact_store_ref="store://artifacts/1",
        trust_record=decision.trust_record,
    )
    assert verified.value.installation_state is InstallationState.VERIFIED


def test_policy_fingerprint_is_deterministic() -> None:
    first = _production_policy()
    second = _production_policy()
    assert first.policy_fingerprint == second.policy_fingerprint
    assert first.policy_fingerprint.startswith("sha256:")


def test_policy_difference_changes_fingerprint() -> None:
    production = _production_policy()
    development = _development_policy()
    assert production.policy_fingerprint != development.policy_fingerprint


def test_allow_decision_records_policy_fingerprint() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None
    assert (
        decision.trust_record.policy_fingerprint
        == _production_policy().policy_fingerprint
    )
    audit = decision.to_audit_dict()
    assert audit["policy_fingerprint"] == decision.trust_record.policy_fingerprint


def test_stale_allow_blocked_after_revocation_at_admission() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None
    coordinator = AgentPackageTrustCoordinator()
    revocation = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset({_DIGEST_A}),
    )
    with pytest.raises(AgentPackageTrustError) as exc_info:
        coordinator.assert_install_admission(
            trust_record=decision.trust_record,
            package_identity=_PACKAGE,
            revocation_state=revocation,
        )
    assert (
        exc_info.value.reason_code
        == AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED.value
    )


def test_revocation_overrides_production_qualification_at_admission() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None
    coordinator = AgentPackageTrustCoordinator()
    with pytest.raises(AgentPackageTrustError) as exc_info:
        coordinator.assert_install_admission(
            trust_record=decision.trust_record,
            package_identity=_PACKAGE,
            revocation_state=AgentPackageTrustRevocationState(
                revoked_evidence_ids=frozenset({"evidence:pkg-a:0"}),
            ),
        )
    assert (
        exc_info.value.reason_code == AgentPackageTrustReasonCode.EVIDENCE_REVOKED.value
    )


def test_source_qualified_evidence_cannot_authorize_other_delivery_source() -> None:
    decision = _evaluate(
        delivery_source=AgentDeliverySource.BUILTIN,
        qualification=_qualification(delivery_source=AgentDeliverySource.MARKETPLACE),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_PACKAGE_MISMATCH


def test_fresh_qualification_continues_normal_evaluation() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=6),
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW


def test_exact_max_age_boundary_is_fresh() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=7),
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW


def test_expired_qualification_denied() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=7, microseconds=1),
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.QUALIFICATION_EXPIRED


def test_future_qualification_timestamp_denied() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(
            qualified_at=_QUALIFIED_AT + timedelta(hours=1),
        ),
        evaluated_at=_QUALIFIED_AT,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert (
        decision.reason_code is AgentPackageTrustReasonCode.QUALIFICATION_TIMESTAMP_INVALID
    )


def test_no_max_age_preserves_backward_compatibility() -> None:
    decision = _evaluate(
        qualification=_qualification(
            qualified_at=_QUALIFIED_AT - timedelta(days=365),
        ),
        evaluated_at=_FIXED_AT,
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW


def test_naive_qualification_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC datetime"):
        AgentPackageQualificationResult(
            publisher=_PUBLISHER,
            status=QualificationStatus.PRODUCTION_QUALIFIED,
            evidence=(),
            reason="naive timestamp",
            delivery_source=AgentDeliverySource.BUILTIN,
            qualified_at=datetime(2026, 8, 6, 12, 0, 0),
        )


def test_requalification_replaces_immutable_snapshot() -> None:
    expired = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=8),
    )
    assert expired.reason_code is AgentPackageTrustReasonCode.QUALIFICATION_EXPIRED

    refreshed = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_FIXED_AT),
        evaluated_at=_FIXED_AT,
    )
    assert refreshed.outcome is AgentPackageTrustOutcome.ALLOW
    assert expired.qualification is not refreshed.qualification
    assert expired.qualification.qualified_at == _QUALIFIED_AT


def test_policy_fingerprint_includes_max_qualification_age() -> None:
    seven_days = _production_policy(max_qualification_age=timedelta(days=7))
    thirty_days = _production_policy(max_qualification_age=timedelta(days=30))
    assert seven_days.policy_fingerprint != thirty_days.policy_fingerprint


def test_stale_qualification_install_replay_blocked() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT,
    )
    assert decision.trust_record is not None
    coordinator = AgentPackageTrustCoordinator()
    with pytest.raises(AgentPackageTrustError) as exc_info:
        coordinator.assert_install_admission(
            trust_record=decision.trust_record,
            package_identity=_PACKAGE,
            policy=_production_policy(max_qualification_age=timedelta(days=7)),
            evaluated_at=_QUALIFIED_AT + timedelta(days=10),
        )
    assert (
        exc_info.value.reason_code
        == AgentPackageTrustReasonCode.QUALIFICATION_EXPIRED.value
    )


def test_policy_tightening_denies_new_admission() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=30)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=10),
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW
    coordinator = AgentPackageTrustCoordinator()
    with pytest.raises(AgentPackageTrustError) as exc_info:
        coordinator.assert_install_admission(
            trust_record=decision.trust_record,
            package_identity=_PACKAGE,
            policy=_production_policy(max_qualification_age=timedelta(days=7)),
            evaluated_at=_QUALIFIED_AT + timedelta(days=10),
        )
    assert (
        exc_info.value.reason_code
        == AgentPackageTrustReasonCode.QUALIFICATION_EXPIRED.value
    )


def test_revocation_precedes_qualification_expiry() -> None:
    decision = _evaluate(
        policy=_production_policy(max_qualification_age=timedelta(days=7)),
        qualification=_qualification(qualified_at=_QUALIFIED_AT),
        evaluated_at=_QUALIFIED_AT + timedelta(days=10),
        revocation_state=AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({_DIGEST_A}),
        ),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED


def test_allow_decision_records_qualification_qualified_at() -> None:
    decision = _evaluate()
    assert decision.trust_record is not None
    assert decision.trust_record.qualification_qualified_at == _QUALIFIED_AT
