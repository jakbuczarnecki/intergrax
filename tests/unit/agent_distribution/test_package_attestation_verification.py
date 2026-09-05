# © Artur Czarnecki. All rights reserved.

"""AC-6 Phase 2 package attestation verification tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.catalog import (
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.errors import AgentPackageAttestationError
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationAlgorithm,
    AgentPackageAttestationStatement,
    AgentPackageAttestationVerificationOutcome,
    AgentPackageAttestationVerificationReasonCode,
    AgentPackageAttestationVerificationRequest,
    StaticPublisherVerificationKeyProvider,
    decode_attestation_signature,
    is_verified_signature_qualification_evidence,
    qualification_evidence_from_attestation_verification,
)
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentPackageQualificationResult,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from testing_support.agent_package_attestation import (
    build_test_attestation_keypair,
    sign_package_attestation_statement,
    verified_signature_qualification_evidence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_PUBLISHER_A = "publisher:acme"
_PUBLISHER_B = "publisher:other"
_KEY_ID = "publisher-key-1"
_PACKAGE = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
)
_PUBLISHER = AgentPublisherIdentity(publisher_id=_PUBLISHER_A, display_name="ACME")
_SOURCE = CatalogSourceIdentity(
    catalog_source_id="builtin",
    provider_kind=CatalogProviderKind.BUILTIN,
)


def _verification_request(
    *,
    package_identity: AgentPackageIdentity = _PACKAGE,
    publisher_id: str = _PUBLISHER_A,
    signature_b64: str,
    public_key_bytes: bytes,
    algorithm: AgentPackageAttestationAlgorithm = AgentPackageAttestationAlgorithm.ED25519,
) -> AgentPackageAttestationVerificationRequest:
    return AgentPackageAttestationVerificationRequest(
        package_identity=package_identity,
        publisher_id=publisher_id,
        attestation_id="attest-1",
        key_id=_KEY_ID,
        algorithm=algorithm,
        signature_b64=signature_b64,
        public_key_bytes=public_key_bytes,
    )


def test_attestation_statement_canonicalization_is_deterministic() -> None:
    first = AgentPackageAttestationStatement.from_package_identity(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
    )
    second = AgentPackageAttestationStatement.from_package_identity(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
    )
    assert first.canonical_bytes() == second.canonical_bytes()


def test_attestation_happy_path_verifies_and_emits_qualification_evidence() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            signature_b64=signature_b64,
            public_key_bytes=public_key_bytes,
        )
    )
    assert result.outcome is AgentPackageAttestationVerificationOutcome.VERIFIED
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.VERIFIED
    assert result.key_id == _KEY_ID
    assert result.algorithm is AgentPackageAttestationAlgorithm.ED25519

    evidence = qualification_evidence_from_attestation_verification(result)
    assert evidence.kind is AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION
    assert is_verified_signature_qualification_evidence(
        evidence,
        expected_package_digest=_DIGEST_A,
    )


def test_attestation_wrong_digest_fails_before_trust() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            package_identity=_PACKAGE.model_copy(update={"package_digest": _DIGEST_B}),
            signature_b64=signature_b64,
            public_key_bytes=public_key_bytes,
        )
    )
    assert result.outcome is AgentPackageAttestationVerificationOutcome.INVALID
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_version", "9.9.9"),
        ("package_digest", _DIGEST_B),
        ("publisher_id", _PUBLISHER_B),
    ],
)
def test_attestation_tampered_statement_fields_fail(field: str, value: str) -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    tampered_identity = _PACKAGE
    publisher_id = _PUBLISHER_A
    if field == "publisher_id":
        publisher_id = value
    else:
        tampered_identity = _PACKAGE.model_copy(update={field: value})
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            package_identity=tampered_identity,
            publisher_id=publisher_id,
            signature_b64=signature_b64,
            public_key_bytes=public_key_bytes,
        )
    )
    assert result.outcome is AgentPackageAttestationVerificationOutcome.INVALID
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE


def test_attestation_wrong_key_fails_closed() -> None:
    private_key, _ = build_test_attestation_keypair()
    _, other_public = build_test_attestation_keypair(seed=b"\x03" * 32)
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            signature_b64=signature_b64,
            public_key_bytes=other_public,
        )
    )
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE


def test_attestation_wrong_publisher_claim_fails_closed() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            publisher_id=_PUBLISHER_B,
            signature_b64=signature_b64,
            public_key_bytes=public_key_bytes,
        )
    )
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE


def test_attestation_malformed_signature_fails_closed() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    verifier = Ed25519PackageAttestationVerifier()
    result = verifier.verify(
        _verification_request(
            signature_b64="not-valid-base64!!!",
            public_key_bytes=public_key_bytes,
        )
    )
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION
    with pytest.raises(AgentPackageAttestationError):
        decode_attestation_signature("%%%invalid%%%")


def test_attestation_unsupported_algorithm_fails_closed() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    verifier = Ed25519PackageAttestationVerifier()

    class _UnsupportedAlgorithm:
        value = "RSA-PSS"

    result = verifier.verify(
        AgentPackageAttestationVerificationRequest(
            package_identity=_PACKAGE,
            publisher_id=_PUBLISHER_A,
            attestation_id="attest-1",
            key_id=_KEY_ID,
            algorithm=_UnsupportedAlgorithm(),  # type: ignore[arg-type]
            signature_b64=signature_b64,
            public_key_bytes=public_key_bytes,
        )
    )
    assert (
        result.reason_code
        is AgentPackageAttestationVerificationReasonCode.UNSUPPORTED_ALGORITHM
    )


def test_attestation_key_provider_resolves_publisher_key() -> None:
    private_key, public_key_bytes = build_test_attestation_keypair()
    signature_b64 = sign_package_attestation_statement(
        package_identity=_PACKAGE,
        publisher_id=_PUBLISHER_A,
        key_id=_KEY_ID,
        private_key=private_key,
    )
    provider = StaticPublisherVerificationKeyProvider(
        {(_PUBLISHER_A, _KEY_ID): public_key_bytes}
    )
    verifier = Ed25519PackageAttestationVerifier(key_provider=provider)
    result = verifier.verify(
        AgentPackageAttestationVerificationRequest(
            package_identity=_PACKAGE,
            publisher_id=_PUBLISHER_A,
            attestation_id="attest-1",
            key_id=_KEY_ID,
            algorithm=AgentPackageAttestationAlgorithm.ED25519,
            signature_b64=signature_b64,
        )
    )
    assert result.verified


def test_forged_signature_qualification_evidence_rejected_by_trust_coordinator() -> None:
    coordinator = AgentPackageTrustCoordinator()
    forged = AgentPackageQualificationResult(
        publisher=_PUBLISHER,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                code="signature_ok",
                ref="sig-ref",
            ),
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="forged",
        delivery_source=AgentDeliverySource.BUILTIN,
    )
    decision = coordinator.evaluate(
        package_identity=_PACKAGE,
        catalog_source=_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=AgentPackageTrustPolicy(
            posture=AgentPackageTrustPosture.PRODUCTION,
            required_evidence_kinds=frozenset(
                {
                    AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    AgentQualificationEvidenceKind.REVOCATION_CHECK,
                }
            ),
        ),
        qualification=forged,
        evidence_package_digest=_DIGEST_A,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.MALFORMED_EVIDENCE


def test_trust_accepts_verified_signature_evidence_when_policy_requires_it() -> None:
    coordinator = AgentPackageTrustCoordinator()
    qualification = AgentPackageQualificationResult(
        publisher=_PUBLISHER,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            verified_signature_qualification_evidence(
                package_identity=_PACKAGE,
                publisher_id=_PUBLISHER_A,
            ),
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="verified",
        delivery_source=AgentDeliverySource.BUILTIN,
    )
    decision = coordinator.evaluate(
        package_identity=_PACKAGE,
        catalog_source=_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=AgentPackageTrustPolicy(
            posture=AgentPackageTrustPosture.PRODUCTION,
            required_evidence_kinds=frozenset(
                {
                    AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    AgentQualificationEvidenceKind.REVOCATION_CHECK,
                }
            ),
        ),
        qualification=qualification,
        evidence_package_digest=_DIGEST_A,
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW


def test_valid_signature_still_denied_when_digest_revoked() -> None:
    coordinator = AgentPackageTrustCoordinator()
    qualification = AgentPackageQualificationResult(
        publisher=_PUBLISHER,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            verified_signature_qualification_evidence(
                package_identity=_PACKAGE,
                publisher_id=_PUBLISHER_A,
            ),
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="verified",
        delivery_source=AgentDeliverySource.BUILTIN,
    )
    decision = coordinator.evaluate(
        package_identity=_PACKAGE,
        catalog_source=_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=AgentPackageTrustPolicy(
            posture=AgentPackageTrustPosture.PRODUCTION,
            required_evidence_kinds=frozenset(
                {
                    AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    AgentQualificationEvidenceKind.REVOCATION_CHECK,
                }
            ),
        ),
        qualification=qualification,
        evidence_package_digest=_DIGEST_A,
        revocation_state=AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({_DIGEST_A})
        ),
    )
    assert decision.reason_code is AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED
