# © Artur Czarnecki. All rights reserved.

"""AP-5 AgentPackageTrust coordinator tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.agent_distribution.catalog import CatalogProviderKind, CatalogSourceIdentity
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.identity import AgentPackageCandidate, AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
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
    AgentQualificationEvidence,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_FIXED_AT = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)

_PACKAGE = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
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
        "required_qualification_status": AgentQualificationStatus.QUALIFIED,
    }
    base.update(overrides)
    return AgentPackageTrustPolicy(**base)


def _qualification(
    *,
    status: AgentQualificationStatus = AgentQualificationStatus.PRODUCTION_QUALIFIED,
    publisher: AgentPublisherIdentity = _PUBLISHER,
    delivery_source: AgentDeliverySource = AgentDeliverySource.BUILTIN,
) -> AgentPackageQualificationResult:
    return AgentPackageQualificationResult(
        publisher=publisher,
        status=status,
        evidence=(
            AgentQualificationEvidence(
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                code="signature_ok",
                ref="sig-ref",
            ),
            AgentQualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="qualified by test evidence",
        delivery_source=delivery_source,
    )


def _evaluate(**overrides: object):
    coordinator = AgentPackageTrustCoordinator()
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
    assert decision.trust_record.qualification_status is AgentQualificationStatus.PRODUCTION_QUALIFIED
    assert decision.trust_record.publisher_identity_ref == "publisher:acme"
    assert len(decision.trust_evidence_refs) == 2


def test_trust_evidence_digest_mismatch_fails_closed() -> None:
    decision = _evaluate(evidence_package_digest=_DIGEST_B)
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH


def test_trust_evidence_for_another_package_fails_closed() -> None:
    decision = _evaluate(evidence_package_digest=_DIGEST_B)
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
    assert decision.reason_code is AgentPackageTrustReasonCode.VERSION_LABEL_WITHOUT_DIGEST


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
        status=AgentQualificationStatus.QUALIFIED,
        evidence=(
            AgentQualificationEvidence(
                kind=AgentQualificationEvidenceKind.CONTRACT_VALIDATION,
                code="contract_ok",
            ),
        ),
        reason="development qualification",
        delivery_source=AgentDeliverySource.LOCAL_DEVELOPER,
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
        qualification=_qualification(status=AgentQualificationStatus.QUALIFIED),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.INSUFFICIENT_QUALIFICATION_STATUS


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
                qualification_status=AgentQualificationStatus.NOT_QUALIFIED,
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
