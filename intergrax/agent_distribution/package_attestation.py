# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cryptographic package attestation contracts (AC-6 Phase 2)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.errors import AgentPackageAttestationError
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.trust import (
    AgentQualificationEvidenceKind,
)
from intergrax.core.qualification import QualificationEvidence
from intergrax.runtime.attestation.canonical_json import canonical_json_bytes

SCHEMA_AGENT_PACKAGE_ATTESTATION_STATEMENT_V1: Final = (
    "agent_package_attestation_statement.v1"
)
SIGNATURE_VERIFICATION_VERIFIED_CODE: Final = "attestation_verified"
SIGNATURE_VERIFICATION_REF_PREFIX: Final = "agent_package_attestation:"
VERIFIER_IMPLEMENTATION_ED25519_V1: Final = "ed25519_package_attestation_verifier.v1"

_ATTESTATION_REF_RE = re.compile(
    r"^agent_package_attestation:(?P<attestation_id>[^;]+);"
    r"key=(?P<key_id>[^;]+);"
    r"alg=(?P<algorithm>[^;]+);"
    r"digest=(?P<digest>sha256:[a-f0-9]{64})$"
)


class AgentPackageAttestationAlgorithm(StrEnum):
    """Supported package attestation signature algorithms."""

    ED25519 = "ED25519"


class AgentPackageAttestationVerificationOutcome(StrEnum):
    """Cryptographic verification outcome — not a trust decision."""

    VERIFIED = "verified"
    INVALID = "invalid"


class AgentPackageAttestationVerificationReasonCode(StrEnum):
    """Stable machine-readable attestation verification reason codes."""

    VERIFIED = "verified"
    INVALID_SIGNATURE = "invalid_signature"
    DIGEST_MISMATCH = "digest_mismatch"
    PUBLISHER_MISMATCH = "publisher_mismatch"
    UNKNOWN_KEY = "unknown_key"
    UNSUPPORTED_ALGORITHM = "unsupported_algorithm"
    MALFORMED_ATTESTATION = "malformed_attestation"


@dataclass(frozen=True, slots=True)
class AgentPackageAttestationStatement:
    """Deterministic signed statement binding artifact digest to publisher identity."""

    schema_id: str
    distribution_package_id: str
    package_version: str
    package_digest: str
    publisher_id: str
    key_id: str

    def canonical_payload(self) -> dict[str, str]:
        return {
            "schema_id": self.schema_id,
            "distribution_package_id": self.distribution_package_id,
            "package_digest": self.package_digest,
            "package_version": self.package_version,
            "publisher_id": self.publisher_id,
            "key_id": self.key_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())

    @classmethod
    def from_package_identity(
        cls,
        *,
        package_identity: AgentPackageIdentity,
        publisher_id: str,
        key_id: str,
    ) -> AgentPackageAttestationStatement:
        return cls(
            schema_id=SCHEMA_AGENT_PACKAGE_ATTESTATION_STATEMENT_V1,
            distribution_package_id=package_identity.distribution_package_id,
            package_version=package_identity.package_version,
            package_digest=package_identity.package_digest,
            publisher_id=publisher_id,
            key_id=key_id,
        )


@dataclass(frozen=True, slots=True)
class AgentPackageAttestationVerificationRequest:
    """Canonical authority inputs for offline package attestation verification."""

    package_identity: AgentPackageIdentity
    publisher_id: str
    attestation_id: str
    key_id: str
    algorithm: AgentPackageAttestationAlgorithm
    signature_b64: str
    public_key_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class AgentPackageAttestationVerificationResult:
    """Diagnostic cryptographic verification report — not qualification authority."""

    outcome: AgentPackageAttestationVerificationOutcome
    reason_code: AgentPackageAttestationVerificationReasonCode
    reason: str
    package_digest: str
    publisher_id: str
    key_id: str
    algorithm: AgentPackageAttestationAlgorithm
    attestation_id: str
    verifier_implementation_id: str

    @property
    def verified(self) -> bool:
        return self.outcome is AgentPackageAttestationVerificationOutcome.VERIFIED


@dataclass(frozen=True, slots=True)
class AgentPackageAttestationQualificationEvidence:
    """Platform-issued signature qualification evidence bound to attestation material."""

    package_digest: str
    publisher_id: str
    attestation_id: str
    key_id: str
    algorithm: AgentPackageAttestationAlgorithm
    signature_b64: str
    verifier_implementation_id: str

    @property
    def kind(self) -> AgentQualificationEvidenceKind:
        return AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION

    @property
    def code(self) -> str:
        return SIGNATURE_VERIFICATION_VERIFIED_CODE

    @property
    def ref(self) -> str:
        return format_attestation_evidence_ref_fields(
            attestation_id=self.attestation_id,
            key_id=self.key_id,
            algorithm=self.algorithm,
            package_digest=self.package_digest,
        )

    @property
    def label(self) -> str:
        return (
            f"{self.algorithm.value} attestation {self.attestation_id} "
            f"via {self.verifier_implementation_id}"
        )

    def to_verification_request(
        self,
        *,
        package_identity: AgentPackageIdentity,
    ) -> AgentPackageAttestationVerificationRequest:
        return AgentPackageAttestationVerificationRequest(
            package_identity=package_identity,
            publisher_id=self.publisher_id,
            attestation_id=self.attestation_id,
            key_id=self.key_id,
            algorithm=self.algorithm,
            signature_b64=self.signature_b64,
        )


@runtime_checkable
class AgentPublisherVerificationKeyProvider(Protocol):
    """Resolve publisher verification keys without network I/O in core verification."""

    def resolve_verification_key(
        self,
        publisher_id: str,
        key_id: str,
    ) -> bytes | None:
        """Return raw Ed25519 public key bytes, or None when unknown."""
        ...


@runtime_checkable
class AgentPackageAttestationVerifier(Protocol):
    """Offline package authenticity verifier — does not decide trust policy."""

    def verify(
        self,
        request: AgentPackageAttestationVerificationRequest,
    ) -> AgentPackageAttestationVerificationResult:
        """Verify cryptographic attestation for the exact digest-pinned package."""
        ...

    def verify_qualification_evidence(
        self,
        request: AgentPackageAttestationVerificationRequest,
    ) -> AgentPackageAttestationQualificationEvidence:
        """Verify attestation and emit canonical SIGNATURE_VERIFICATION evidence."""
        ...


class StaticPublisherVerificationKeyProvider:
    """Pinned publisher verification keys for Reference Production V1."""

    def __init__(self, keys: dict[tuple[str, str], bytes]) -> None:
        self._keys = dict(keys)

    def resolve_verification_key(
        self,
        publisher_id: str,
        key_id: str,
    ) -> bytes | None:
        return self._keys.get((publisher_id.strip(), key_id.strip()))


def decode_attestation_signature(signature_b64: str) -> bytes:
    """Decode base64 signature bytes or raise AgentPackageAttestationError."""
    normalized = signature_b64.strip()
    if not normalized:
        raise AgentPackageAttestationError(
            "attestation signature must be non-empty",
            reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION.value,
        )
    try:
        signature = base64.b64decode(normalized.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise AgentPackageAttestationError(
            "attestation signature is not valid base64",
            reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION.value,
        ) from exc
    if len(signature) != 64:
        raise AgentPackageAttestationError(
            "Ed25519 signature must be 64 bytes",
            reason_code=AgentPackageAttestationVerificationReasonCode.MALFORMED_ATTESTATION.value,
        )
    return signature


def format_attestation_evidence_ref_fields(
    *,
    attestation_id: str,
    key_id: str,
    algorithm: AgentPackageAttestationAlgorithm,
    package_digest: str,
) -> str:
    digest = normalize_package_digest(package_digest)
    return (
        f"{SIGNATURE_VERIFICATION_REF_PREFIX}{attestation_id}"
        f";key={key_id};alg={algorithm.value};digest={digest}"
    )


def format_attestation_evidence_ref(
    result: AgentPackageAttestationVerificationResult,
) -> str:
    return format_attestation_evidence_ref_fields(
        attestation_id=result.attestation_id,
        key_id=result.key_id,
        algorithm=result.algorithm,
        package_digest=result.package_digest,
    )


def parse_attestation_evidence_ref(ref: str) -> dict[str, str] | None:
    match = _ATTESTATION_REF_RE.match(ref.strip())
    if match is None:
        return None
    return match.groupdict()


def is_verified_signature_qualification_evidence(
    evidence: QualificationEvidence[AgentQualificationEvidenceKind]
    | AgentPackageAttestationQualificationEvidence,
    *,
    expected_package_digest: str | None = None,
) -> bool:
    """Return True only for structurally valid platform attestation evidence."""
    if isinstance(evidence, AgentPackageAttestationQualificationEvidence):
        if evidence.code != SIGNATURE_VERIFICATION_VERIFIED_CODE:
            return False
        if parse_attestation_evidence_ref(evidence.ref) is None:
            return False
        digest = evidence.package_digest
    else:
        return False

    if expected_package_digest is not None:
        try:
            expected = normalize_package_digest(expected_package_digest)
            actual = normalize_package_digest(digest)
        except ValueError:
            return False
        if expected != actual:
            return False
    return True
