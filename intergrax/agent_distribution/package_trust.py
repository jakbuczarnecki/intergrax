# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent package trust coordination (AGENT_DISTRIBUTION §10, AP-5)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.catalog import CatalogSourceIdentity
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.identity import (
    AgentPackageCandidate,
    AgentPackageIdentity,
)
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationQualificationEvidence,
    AgentPackageAttestationVerifier,
    is_verified_signature_qualification_evidence,
)
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentInstallationTrustRecord,
    AgentPackageQualificationResult,
    AgentPackageTrustDecision,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.core.qualification import (
    QualificationEvidence,
    QualificationStatus,
    qualification_status_satisfies,
)


class AgentPackageTrustCoordinator:
    """Fail-closed trust and qualification gate for digest-pinned agent packages."""

    def __init__(
        self,
        *,
        attestation_verifier: AgentPackageAttestationVerifier | None = None,
    ) -> None:
        self._attestation_verifier = attestation_verifier

    def evaluate(
        self,
        *,
        package_identity: AgentPackageIdentity,
        catalog_source: CatalogSourceIdentity,
        delivery_source: AgentDeliverySource,
        publisher: AgentPublisherIdentity,
        policy: AgentPackageTrustPolicy,
        qualification: AgentPackageQualificationResult | None = None,
        evidence_package_digest: str | None = None,
        evidence_id: str | None = None,
        source_entry_ref: str | None = None,
        org_policy_decision_ref: str | None = None,
        revocation_state: AgentPackageTrustRevocationState | None = None,
        evaluated_at: datetime | None = None,
    ) -> AgentPackageTrustDecision:
        """Evaluate whether ``package_identity`` may be trusted under ``policy``."""
        revocation = revocation_state or AgentPackageTrustRevocationState()
        checked_at = evaluated_at or datetime.now(UTC)

        digest = package_identity.package_digest
        publisher_ref = publisher.publisher_id
        source_id = catalog_source.catalog_source_id

        if digest in policy.denied_package_digests:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED,
                reason="package digest is explicitly denied by policy",
            )

        if digest in revocation.revoked_package_digests:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED,
                reason="package digest is revoked",
            )

        if publisher_ref in policy.denied_publisher_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PUBLISHER_DENIED,
                reason="publisher is explicitly denied by policy",
            )

        if publisher_ref in revocation.revoked_publisher_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PUBLISHER_REVOKED,
                reason="publisher is revoked",
            )

        if source_id in policy.denied_catalog_source_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.SOURCE_DENIED,
                reason="catalog source is explicitly denied by policy",
            )

        if source_id in revocation.revoked_catalog_source_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.SOURCE_REVOKED,
                reason="catalog source is revoked",
            )

        if source_id in revocation.disabled_catalog_source_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.SOURCE_DISABLED,
                reason="catalog source is disabled",
            )

        source_denial = self._evaluate_source_policy(
            package_identity=package_identity,
            catalog_source=catalog_source,
            delivery_source=delivery_source,
            publisher=publisher,
            policy=policy,
            qualification=qualification,
        )
        if source_denial is not None:
            return source_denial

        if (
            policy.permitted_publisher_ids is not None
            and publisher_ref not in policy.permitted_publisher_ids
        ):
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PUBLISHER_DENIED,
                reason="publisher is not permitted by policy",
            )

        if evidence_id is not None and evidence_id in revocation.revoked_evidence_ids:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.EVIDENCE_REVOKED,
                reason="qualification evidence is revoked",
            )

        if qualification is None:
            if policy.forbid_unsigned_or_unqualified:
                return self._deny(
                    package_identity=package_identity,
                    publisher=publisher,
                    catalog_source_id=source_id,
                    delivery_source=delivery_source,
                    policy=policy,
                    qualification=None,
                    reason_code=AgentPackageTrustReasonCode.MISSING_REQUIRED_EVIDENCE,
                    reason="required qualification evidence is missing",
                )
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=None,
                reason_code=AgentPackageTrustReasonCode.UNQUALIFIED_FORBIDDEN,
                reason="package is not qualified",
            )

        if qualification.publisher.publisher_id != publisher_ref:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.PUBLISHER_MISMATCH,
                reason="qualification publisher does not match supplied publisher",
            )

        if qualification.delivery_source is not delivery_source:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.EVIDENCE_PACKAGE_MISMATCH,
                reason="qualification delivery source does not match evaluation input",
            )

        if qualification.status is QualificationStatus.REJECTED:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.INSUFFICIENT_QUALIFICATION_STATUS,
                reason="qualification evidence is rejected",
            )

        required_status = policy.effective_required_qualification_status
        if not qualification_status_satisfies(qualification.status, required_status):
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.INSUFFICIENT_QUALIFICATION_STATUS,
                reason=(
                    "qualification status does not satisfy policy requirement "
                    f"{required_status.value}"
                ),
            )

        present_kinds = {item.kind for item in qualification.evidence}
        missing_kinds = policy.required_evidence_kinds - present_kinds
        if missing_kinds:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.MISSING_REQUIRED_EVIDENCE,
                reason=(
                    "required qualification evidence kinds are missing: "
                    + ", ".join(sorted(kind.value for kind in missing_kinds))
                ),
            )

        if not qualification.evidence:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                reason="qualification evidence payload is empty",
            )

        if evidence_package_digest is None:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.MISSING_PACKAGE_DIGEST_EVIDENCE,
                reason="qualification evidence is not bound to a package digest",
            )

        try:
            evidence_digest = normalize_package_digest(evidence_package_digest)
        except ValueError:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                reason="qualification evidence package digest is malformed",
            )

        if evidence_digest != digest:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH,
                reason="qualification evidence digest does not match package digest",
            )

        signature_validation = self._validate_signature_verification_evidence(
            qualification.evidence,
            required_kinds=policy.required_evidence_kinds,
            package_digest=digest,
            package_identity=package_identity,
            publisher=publisher,
            catalog_source_id=source_id,
            delivery_source=delivery_source,
            policy=policy,
            qualification=qualification,
        )
        if signature_validation is not None:
            return signature_validation

        trust_evidence_refs = self._build_evidence_refs(
            qualification.evidence,
            evidence_id=evidence_id,
        )
        trust_record = AgentInstallationTrustRecord(
            trust_evidence_refs=trust_evidence_refs,
            qualification_status=qualification.status,
            package_digest=digest,
            publisher_identity_ref=publisher_ref,
            source_provider_id=source_id,
            source_entry_ref=source_entry_ref,
            revocation_checked_at=checked_at,
            org_policy_decision_ref=org_policy_decision_ref,
            policy_fingerprint=policy.policy_fingerprint,
        )

        return AgentPackageTrustDecision(
            outcome=AgentPackageTrustOutcome.ALLOW,
            reason_code=AgentPackageTrustReasonCode.QUALIFIED,
            reason="package qualification evidence satisfies trust policy",
            package_identity=package_identity,
            publisher=publisher,
            catalog_source_id=source_id,
            delivery_source=delivery_source,
            policy_profile_ref=policy.trust_profile_ref,
            qualification=qualification,
            trust_record=trust_record,
            trust_evidence_refs=trust_evidence_refs,
        )

    def evaluate_candidate(
        self,
        *,
        package_candidate: AgentPackageCandidate,
        catalog_source: CatalogSourceIdentity,
        delivery_source: AgentDeliverySource,
        publisher: AgentPublisherIdentity,
        policy: AgentPackageTrustPolicy,
        qualification: AgentPackageQualificationResult | None = None,
        evidence_package_digest: str | None = None,
        evidence_id: str | None = None,
        source_entry_ref: str | None = None,
        org_policy_decision_ref: str | None = None,
        revocation_state: AgentPackageTrustRevocationState | None = None,
        evaluated_at: datetime | None = None,
    ) -> AgentPackageTrustDecision:
        """Evaluate a pre-verification candidate — version labels alone cannot authorize."""
        if package_candidate.package_digest is None:
            return self._deny(
                package_identity=AgentPackageIdentity(
                    distribution_package_id=package_candidate.distribution_package_id,
                    package_version=package_candidate.package_version,
                    package_digest=evidence_package_digest or ("sha256:" + ("0" * 64)),
                ),
                publisher=publisher,
                catalog_source_id=catalog_source.catalog_source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.VERSION_LABEL_WITHOUT_DIGEST,
                reason="version label without digest cannot authorize qualification",
            )
        return self.evaluate(
            package_identity=package_candidate.to_digest_pinned(),
            catalog_source=catalog_source,
            delivery_source=delivery_source,
            publisher=publisher,
            policy=policy,
            qualification=qualification,
            evidence_package_digest=evidence_package_digest,
            evidence_id=evidence_id,
            source_entry_ref=source_entry_ref,
            org_policy_decision_ref=org_policy_decision_ref,
            revocation_state=revocation_state,
            evaluated_at=evaluated_at,
        )

    def _evaluate_source_policy(
        self,
        *,
        package_identity: AgentPackageIdentity,
        catalog_source: CatalogSourceIdentity,
        delivery_source: AgentDeliverySource,
        publisher: AgentPublisherIdentity,
        policy: AgentPackageTrustPolicy,
        qualification: AgentPackageQualificationResult | None,
    ) -> AgentPackageTrustDecision | None:
        source_id = catalog_source.catalog_source_id
        common = {
            "package_identity": package_identity,
            "publisher": publisher,
            "catalog_source_id": source_id,
            "delivery_source": delivery_source,
            "policy": policy,
            "qualification": qualification,
        }

        if (
            policy.permitted_provider_kinds is not None
            and catalog_source.provider_kind not in policy.permitted_provider_kinds
        ):
            return self._deny(
                **common,
                reason_code=AgentPackageTrustReasonCode.SOURCE_NOT_PERMITTED,
                reason="catalog provider kind is not permitted by policy",
            )

        if (
            policy.permitted_catalog_source_ids is not None
            and source_id not in policy.permitted_catalog_source_ids
        ):
            return self._deny(
                **common,
                reason_code=AgentPackageTrustReasonCode.SOURCE_NOT_PERMITTED,
                reason="catalog source is not permitted by policy",
            )

        if (
            policy.permitted_delivery_sources is not None
            and delivery_source not in policy.permitted_delivery_sources
        ):
            return self._deny(
                **common,
                reason_code=AgentPackageTrustReasonCode.SOURCE_NOT_PERMITTED,
                reason="delivery source is not permitted by policy",
            )

        return None

    def assert_install_admission(
        self,
        *,
        trust_record: AgentInstallationTrustRecord,
        package_identity: AgentPackageIdentity,
        revocation_state: AgentPackageTrustRevocationState | None = None,
    ) -> None:
        """Fail closed when stale trust evidence cannot satisfy current revocation state."""
        assert_installation_trust_record_acceptable(
            trust_record,
            package_identity=package_identity,
        )
        revocation = revocation_state or AgentPackageTrustRevocationState()
        digest = package_identity.package_digest

        if digest in revocation.revoked_package_digests:
            raise AgentPackageTrustError(
                "package digest is revoked at install admission",
                reason_code=AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED.value,
            )

        if trust_record.publisher_identity_ref in revocation.revoked_publisher_ids:
            raise AgentPackageTrustError(
                "publisher is revoked at install admission",
                reason_code=AgentPackageTrustReasonCode.PUBLISHER_REVOKED.value,
            )

        source_id = trust_record.source_provider_id
        if source_id in revocation.revoked_catalog_source_ids:
            raise AgentPackageTrustError(
                "catalog source is revoked at install admission",
                reason_code=AgentPackageTrustReasonCode.SOURCE_REVOKED.value,
            )

        if source_id in revocation.disabled_catalog_source_ids:
            raise AgentPackageTrustError(
                "catalog source is disabled at install admission",
                reason_code=AgentPackageTrustReasonCode.SOURCE_DISABLED.value,
            )

        for evidence_ref in trust_record.trust_evidence_refs:
            if evidence_ref.evidence_id in revocation.revoked_evidence_ids:
                raise AgentPackageTrustError(
                    "qualification evidence is revoked at install admission",
                    reason_code=AgentPackageTrustReasonCode.EVIDENCE_REVOKED.value,
                )

    def _validate_signature_verification_evidence(
        self,
        evidence: tuple[QualificationEvidence[AgentQualificationEvidenceKind], ...],
        *,
        required_kinds: frozenset[AgentQualificationEvidenceKind],
        package_digest: str,
        package_identity: AgentPackageIdentity,
        publisher: AgentPublisherIdentity,
        catalog_source_id: str,
        delivery_source: AgentDeliverySource,
        policy: AgentPackageTrustPolicy,
        qualification: AgentPackageQualificationResult,
    ) -> AgentPackageTrustDecision | None:
        del required_kinds
        signature_items = [
            item
            for item in evidence
            if item.kind is AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION
        ]
        if not signature_items:
            return None
        if self._attestation_verifier is None:
            return self._deny(
                package_identity=package_identity,
                publisher=publisher,
                catalog_source_id=catalog_source_id,
                delivery_source=delivery_source,
                policy=policy,
                qualification=qualification,
                reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                reason=(
                    "signature verification evidence requires an injected "
                    "attestation verifier"
                ),
            )
        for item in signature_items:
            if not isinstance(item, AgentPackageAttestationQualificationEvidence):
                return self._deny(
                    package_identity=package_identity,
                    publisher=publisher,
                    catalog_source_id=catalog_source_id,
                    delivery_source=delivery_source,
                    policy=policy,
                    qualification=qualification,
                    reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                    reason=(
                        "signature verification evidence lacks platform-verified "
                        "attestation provenance"
                    ),
                )
            if item.publisher_id != publisher.publisher_id:
                return self._deny(
                    package_identity=package_identity,
                    publisher=publisher,
                    catalog_source_id=catalog_source_id,
                    delivery_source=delivery_source,
                    policy=policy,
                    qualification=qualification,
                    reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                    reason="signature verification evidence publisher mismatch",
                )
            if not is_verified_signature_qualification_evidence(
                item,
                expected_package_digest=package_digest,
            ):
                return self._deny(
                    package_identity=package_identity,
                    publisher=publisher,
                    catalog_source_id=catalog_source_id,
                    delivery_source=delivery_source,
                    policy=policy,
                    qualification=qualification,
                    reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                    reason=(
                        "signature verification evidence lacks platform-verified "
                        "attestation provenance"
                    ),
                )
            verification = self._attestation_verifier.verify(
                item.to_verification_request(package_identity=package_identity)
            )
            if not verification.verified:
                return self._deny(
                    package_identity=package_identity,
                    publisher=publisher,
                    catalog_source_id=catalog_source_id,
                    delivery_source=delivery_source,
                    policy=policy,
                    qualification=qualification,
                    reason_code=AgentPackageTrustReasonCode.MALFORMED_EVIDENCE,
                    reason=(
                        "signature verification evidence failed cryptographic "
                        "re-validation"
                    ),
                )
        return None

    @staticmethod
    def _build_evidence_refs(
        evidence: tuple[QualificationEvidence[AgentQualificationEvidenceKind], ...],
        *,
        evidence_id: str | None,
    ) -> tuple[AgentTrustEvidenceRef, ...]:
        refs: list[AgentTrustEvidenceRef] = []
        for index, item in enumerate(evidence):
            refs.append(
                AgentTrustEvidenceRef(
                    evidence_id=(
                        f"{evidence_id}:{index}"
                        if evidence_id is not None
                        else f"{item.kind.value}:{item.code}:{index}"
                    ),
                    kind=item.kind,
                    ref=item.ref,
                )
            )
        return tuple(refs)

    @staticmethod
    def _deny(
        *,
        package_identity: AgentPackageIdentity,
        publisher: AgentPublisherIdentity,
        catalog_source_id: str,
        delivery_source: AgentDeliverySource,
        policy: AgentPackageTrustPolicy,
        qualification: AgentPackageQualificationResult | None,
        reason_code: AgentPackageTrustReasonCode,
        reason: str,
    ) -> AgentPackageTrustDecision:
        return AgentPackageTrustDecision(
            outcome=AgentPackageTrustOutcome.DENY,
            reason_code=reason_code,
            reason=reason,
            package_identity=package_identity,
            publisher=publisher,
            catalog_source_id=catalog_source_id,
            delivery_source=delivery_source,
            policy_profile_ref=policy.trust_profile_ref,
            qualification=qualification,
            trust_record=None,
            trust_evidence_refs=(),
        )


def assert_installation_trust_record_acceptable(
    trust_record: AgentInstallationTrustRecord,
    *,
    package_identity: AgentPackageIdentity,
) -> None:
    """Validate trust evidence before candidate → verified installation transition."""
    if trust_record.qualification_status in {
        QualificationStatus.NOT_QUALIFIED,
        QualificationStatus.REJECTED,
    }:
        raise AgentPackageTrustError(
            "installation verification requires qualified trust evidence; "
            f"status={trust_record.qualification_status.value}"
        )
    if not qualification_status_satisfies(
        trust_record.qualification_status,
        QualificationStatus.QUALIFIED,
    ):
        raise AgentPackageTrustError(
            "installation verification requires at least qualified trust status"
        )
    if not trust_record.trust_evidence_refs:
        raise AgentPackageTrustError(
            "installation verification requires non-empty trust_evidence_refs"
        )
    if not trust_record.publisher_identity_ref:
        raise AgentPackageTrustError(
            "installation verification requires publisher_identity_ref"
        )
    if not trust_record.source_provider_id:
        raise AgentPackageTrustError(
            "installation verification requires source_provider_id"
        )
    if package_identity.package_digest == "":
        raise AgentPackageTrustError(
            "installation verification requires digest-pinned package"
        )
    if trust_record.package_digest != package_identity.package_digest:
        raise AgentPackageTrustError(
            "installation verification trust record digest does not match package digest"
        )
