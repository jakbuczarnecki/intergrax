# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ed25519 package attestation verifier (AC-6 Phase 2 Reference Production V1)."""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.errors import AgentPackageAttestationError
from intergrax.agent_distribution.package_attestation import (
    VERIFIER_IMPLEMENTATION_ED25519_V1,
    AgentPackageAttestationAlgorithm,
    AgentPackageAttestationQualificationEvidence,
    AgentPackageAttestationStatement,
    AgentPackageAttestationVerificationOutcome,
    AgentPackageAttestationVerificationReasonCode,
    AgentPackageAttestationVerificationRequest,
    AgentPackageAttestationVerificationResult,
    AgentPublisherVerificationKeyProvider,
    decode_attestation_signature,
)


class Ed25519PackageAttestationVerifier:
    """Offline Ed25519 verifier — proves authenticity, not trust policy."""

    def __init__(
        self,
        *,
        key_provider: AgentPublisherVerificationKeyProvider | None = None,
        verifier_implementation_id: str = VERIFIER_IMPLEMENTATION_ED25519_V1,
    ) -> None:
        self._key_provider = key_provider
        self._verifier_implementation_id = verifier_implementation_id

    def verify(
        self,
        request: AgentPackageAttestationVerificationRequest,
    ) -> AgentPackageAttestationVerificationResult:
        common = {
            "package_digest": request.package_identity.package_digest,
            "publisher_id": request.publisher_id,
            "key_id": request.key_id,
            "algorithm": request.algorithm,
            "attestation_id": request.attestation_id,
            "verifier_implementation_id": self._verifier_implementation_id,
        }

        if request.algorithm is not AgentPackageAttestationAlgorithm.ED25519:
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.UNSUPPORTED_ALGORITHM,
                reason=f"unsupported attestation algorithm: {request.algorithm.value}",
                **common,
            )

        try:
            package_digest = normalize_package_digest(
                request.package_identity.package_digest
            )
        except ValueError:
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION,
                reason="package identity digest is malformed",
                **common,
            )

        statement = AgentPackageAttestationStatement.from_package_identity(
            package_identity=request.package_identity.model_copy(
                update={"package_digest": package_digest}
            ),
            publisher_id=request.publisher_id,
            key_id=request.key_id,
        )
        if statement.package_digest != package_digest:
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.DIGEST_MISMATCH,
                reason="attestation statement digest does not match package identity",
                **common,
            )

        public_key_bytes = request.public_key_bytes
        if public_key_bytes is None:
            if self._key_provider is None:
                return self._invalid(
                    reason_code=AgentPackageAttestationVerificationReasonCode.UNKNOWN_KEY,
                    reason="no verification key supplied and no key provider configured",
                    **common,
                )
            public_key_bytes = self._key_provider.resolve_verification_key(
                request.publisher_id,
                request.key_id,
            )
            if public_key_bytes is None:
                return self._invalid(
                    reason_code=AgentPackageAttestationVerificationReasonCode.UNKNOWN_KEY,
                    reason="verification key is unknown for publisher/key_id",
                    **common,
                )

        try:
            signature = decode_attestation_signature(request.signature_b64)
        except Exception:
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION,
                reason="attestation signature is malformed",
                **common,
            )

        try:
            key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            key.verify(signature, statement.canonical_bytes())
        except InvalidSignature:
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE,
                reason="attestation signature is cryptographically invalid",
                **common,
            )
        except (ValueError, TypeError):
            return self._invalid(
                reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION,
                reason="verification key material is malformed",
                **common,
            )

        return AgentPackageAttestationVerificationResult(
            outcome=AgentPackageAttestationVerificationOutcome.VERIFIED,
            reason_code=AgentPackageAttestationVerificationReasonCode.VERIFIED,
            reason="package attestation signature verified",
            package_digest=package_digest,
            publisher_id=request.publisher_id,
            key_id=request.key_id,
            algorithm=request.algorithm,
            attestation_id=request.attestation_id,
            verifier_implementation_id=self._verifier_implementation_id,
        )

    def verify_qualification_evidence(
        self,
        request: AgentPackageAttestationVerificationRequest,
    ) -> AgentPackageAttestationQualificationEvidence:
        result = self.verify(request)
        if not result.verified:
            raise AgentPackageAttestationError(
                "cannot emit signature qualification evidence for invalid attestation",
                reason_code=result.reason_code.value,
            )
        return AgentPackageAttestationQualificationEvidence(
            package_digest=result.package_digest,
            publisher_id=result.publisher_id,
            attestation_id=result.attestation_id,
            key_id=result.key_id,
            algorithm=result.algorithm,
            signature_b64=request.signature_b64,
            verifier_implementation_id=result.verifier_implementation_id,
        )

    @staticmethod
    def _invalid(
        *,
        reason_code: AgentPackageAttestationVerificationReasonCode,
        reason: str,
        package_digest: str,
        publisher_id: str,
        key_id: str,
        algorithm: AgentPackageAttestationAlgorithm,
        attestation_id: str,
        verifier_implementation_id: str,
    ) -> AgentPackageAttestationVerificationResult:
        return AgentPackageAttestationVerificationResult(
            outcome=AgentPackageAttestationVerificationOutcome.INVALID,
            reason_code=reason_code,
            reason=reason,
            package_digest=package_digest,
            publisher_id=publisher_id,
            key_id=key_id,
            algorithm=algorithm,
            attestation_id=attestation_id,
            verifier_implementation_id=verifier_implementation_id,
        )
