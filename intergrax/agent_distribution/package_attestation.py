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
    """Typed immutable cryptographic verification result."""

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


def format_attestation_evidence_ref(
    result: AgentPackageAttestationVerificationResult,
) -> str:
    digest = normalize_package_digest(result.package_digest)
    return (
        f"{SIGNATURE_VERIFICATION_REF_PREFIX}{result.attestation_id}"
        f";key={result.key_id};alg={result.algorithm.value};digest={digest}"
    )


def parse_attestation_evidence_ref(ref: str) -> dict[str, str] | None:
    match = _ATTESTATION_REF_RE.match(ref.strip())
    if match is None:
        return None
    return match.groupdict()


def is_verified_signature_qualification_evidence(
    evidence: QualificationEvidence[AgentQualificationEvidenceKind],
    *,
    expected_package_digest: str | None = None,
) -> bool:
    """Return True only for platform-verified signature qualification evidence."""
    if evidence.kind is not AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION:
        return False
    if evidence.code != SIGNATURE_VERIFICATION_VERIFIED_CODE:
        return False
    if evidence.ref is None:
        return False
    parsed = parse_attestation_evidence_ref(evidence.ref)
    if parsed is None:
        return False
    if expected_package_digest is not None:
        try:
            expected = normalize_package_digest(expected_package_digest)
            actual = normalize_package_digest(parsed["digest"])
        except ValueError:
            return False
        if expected != actual:
            return False
    return True


def qualification_evidence_from_attestation_verification(
    result: AgentPackageAttestationVerificationResult,
) -> QualificationEvidence[AgentQualificationEvidenceKind]:
    """Map a verified attestation result into qualification evidence."""
    if not result.verified:
        raise AgentPackageAttestationError(
            "cannot emit signature qualification evidence for invalid attestation",
            reason_code=result.reason_code.value,
        )
    return QualificationEvidence(
        kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
        code=SIGNATURE_VERIFICATION_VERIFIED_CODE,
        ref=format_attestation_evidence_ref(result),
        label=(
            f"{result.algorithm.value} attestation {result.attestation_id} "
            f"via {result.verifier_implementation_id}"
        ),
    )
