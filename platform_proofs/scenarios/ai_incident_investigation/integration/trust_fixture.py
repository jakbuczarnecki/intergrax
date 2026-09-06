# © Artur Czarnecki. All rights reserved.

"""Canonical AC-6 trust fixture for AIPV-1 incident investigator proofs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.agent_distribution.catalog import (
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentInstallationTrustRecord,
    AgentPackageQualificationResult,
    AgentPackageTrustDecision,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
)
from intergrax.core.qualification import QualificationStatus
from platform_proofs.scenarios.ai_incident_investigation.integration.package_identity import (
    INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
    INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
    INCIDENT_INVESTIGATOR_PUBLISHER_ID,
    incident_investigator_package_identity,
)
from testing_support.agent_package_attestation import (
    build_test_attestation_keypair,
    build_test_attestation_trust_coordinator,
    verified_signature_qualification_evidence,
)

AIPV_QUALIFIED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
AIPV_EVALUATED_AT = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
AIPV_TRUST_EVIDENCE_ID = "evidence:aipv-1"
AIPV_ATTESTATION_ID = "attest-aipv-1"
_PUBLISHER_KEY_ID = "test-publisher-key-1"


def _incident_investigator_publisher() -> AgentPublisherIdentity:
    return AgentPublisherIdentity(publisher_id=INCIDENT_INVESTIGATOR_PUBLISHER_ID)


def _incident_investigator_catalog_source() -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )


def _incident_investigator_trust_policy() -> AgentPackageTrustPolicy:
    return AgentPackageTrustPolicy(
        posture=AgentPackageTrustPosture.PRODUCTION,
        permitted_provider_kinds=frozenset({CatalogProviderKind.BUILTIN}),
        permitted_catalog_source_ids=frozenset(
            {INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID}
        ),
        permitted_delivery_sources=frozenset({AgentDeliverySource.BUILTIN}),
        permitted_publisher_ids=frozenset({INCIDENT_INVESTIGATOR_PUBLISHER_ID}),
        required_qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        required_evidence_kinds=frozenset(
            {AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION}
        ),
    )


def _build_trust_coordinator() -> AgentPackageTrustCoordinator:
    _, public_key_bytes = build_test_attestation_keypair()
    return build_test_attestation_trust_coordinator(
        keys={(INCIDENT_INVESTIGATOR_PUBLISHER_ID, _PUBLISHER_KEY_ID): public_key_bytes}
    )


def _build_qualification(
    package_identity: AgentPackageIdentity,
) -> AgentPackageQualificationResult:
    publisher = _incident_investigator_publisher()
    return AgentPackageQualificationResult(
        publisher=publisher,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            verified_signature_qualification_evidence(
                package_identity=package_identity,
                publisher_id=publisher.publisher_id,
                attestation_id=AIPV_ATTESTATION_ID,
                key_id=_PUBLISHER_KEY_ID,
            ),
        ),
        reason="aipv1_production_validation",
        delivery_source=AgentDeliverySource.BUILTIN,
        qualified_at=AIPV_QUALIFIED_AT,
    )


@dataclass(frozen=True, slots=True)
class IncidentInvestigatorTrustFixture:
    package_identity: AgentPackageIdentity
    publisher: AgentPublisherIdentity
    catalog_source: CatalogSourceIdentity
    delivery_source: AgentDeliverySource
    policy: AgentPackageTrustPolicy
    qualification: AgentPackageQualificationResult
    revocation_state: AgentPackageTrustRevocationState
    evaluated_at: datetime
    coordinator: AgentPackageTrustCoordinator

    @classmethod
    def build(
        cls,
        *,
        revocation_state: AgentPackageTrustRevocationState | None = None,
        evaluated_at: datetime = AIPV_EVALUATED_AT,
    ) -> IncidentInvestigatorTrustFixture:
        package_identity = incident_investigator_package_identity()
        return cls(
            package_identity=package_identity,
            publisher=_incident_investigator_publisher(),
            catalog_source=_incident_investigator_catalog_source(),
            delivery_source=AgentDeliverySource.BUILTIN,
            policy=_incident_investigator_trust_policy(),
            qualification=_build_qualification(package_identity),
            revocation_state=revocation_state or AgentPackageTrustRevocationState(),
            evaluated_at=evaluated_at,
            coordinator=_build_trust_coordinator(),
        )

    def evaluate(
        self,
        *,
        revocation_state: AgentPackageTrustRevocationState | None = None,
    ) -> AgentPackageTrustDecision:
        return self.coordinator.evaluate(
            package_identity=self.package_identity,
            catalog_source=self.catalog_source,
            delivery_source=self.delivery_source,
            publisher=self.publisher,
            policy=self.policy,
            qualification=self.qualification,
            evidence_package_digest=self.package_identity.package_digest,
            evidence_id=AIPV_TRUST_EVIDENCE_ID,
            revocation_state=revocation_state or self.revocation_state,
            evaluated_at=self.evaluated_at,
        )


def evaluate_incident_investigator_trust(
    *,
    revocation_state: AgentPackageTrustRevocationState | None = None,
) -> AgentPackageTrustDecision:
    return IncidentInvestigatorTrustFixture.build(
        revocation_state=revocation_state,
    ).evaluate()


class IncidentInvestigatorCanonicalTrustRecordFactory:
    """Scenario adapter delegating delegated-subtask trust to AC-6 coordinator."""

    def __init__(self, fixture: IncidentInvestigatorTrustFixture | None = None) -> None:
        self._fixture = fixture or IncidentInvestigatorTrustFixture.build()

    def build_trust_record(
        self,
        *,
        package_digest: str,
        package_id: str,
    ) -> AgentInstallationTrustRecord:
        if package_id != INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID:
            msg = f"unexpected package_id for incident investigator trust: {package_id}"
            raise ValueError(msg)
        if package_digest != INCIDENT_INVESTIGATOR_PACKAGE_DIGEST:
            msg = (
                "package_digest does not match incident investigator "
                f"canonical identity: {package_digest}"
            )
            raise ValueError(msg)
        decision = self._fixture.evaluate()
        if decision.outcome is not AgentPackageTrustOutcome.ALLOW:
            msg = f"canonical trust evaluation denied: {decision.reason_code}"
            raise RuntimeError(msg)
        if decision.trust_record is None:
            raise RuntimeError("canonical trust evaluation produced no trust record")
        return decision.trust_record
