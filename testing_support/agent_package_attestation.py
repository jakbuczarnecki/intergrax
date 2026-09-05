# © Artur Czarnecki. All rights reserved.

"""Test helpers for verified package attestation qualification evidence."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationAlgorithm,
    AgentPackageAttestationQualificationEvidence,
    AgentPackageAttestationStatement,
    AgentPackageAttestationVerificationRequest,
    StaticPublisherVerificationKeyProvider,
)
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator

_DEFAULT_SEED = b"\x01" * 32


def build_test_attestation_keypair(
    *,
    seed: bytes = _DEFAULT_SEED,
) -> tuple[Ed25519PrivateKey, bytes]:
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    return private_key, private_key.public_key().public_bytes_raw()


def sign_package_attestation_statement(
    *,
    package_identity: AgentPackageIdentity,
    publisher_id: str,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> str:
    statement = AgentPackageAttestationStatement.from_package_identity(
        package_identity=package_identity,
        publisher_id=publisher_id,
        key_id=key_id,
    )
    signature = private_key.sign(statement.canonical_bytes())
    return base64.b64encode(signature).decode("ascii")


def build_test_attestation_verifier(
    *,
    keys: dict[tuple[str, str], bytes] | None = None,
) -> Ed25519PackageAttestationVerifier:
    if keys is None:
        _, public_key_bytes = build_test_attestation_keypair()
        keys = {("publisher:acme", "test-publisher-key-1"): public_key_bytes}
    return Ed25519PackageAttestationVerifier(
        key_provider=StaticPublisherVerificationKeyProvider(keys)
    )


def build_test_attestation_trust_coordinator(
    *,
    keys: dict[tuple[str, str], bytes] | None = None,
) -> AgentPackageTrustCoordinator:
    return AgentPackageTrustCoordinator(
        attestation_verifier=build_test_attestation_verifier(keys=keys)
    )


def verified_signature_qualification_evidence(
    *,
    package_identity: AgentPackageIdentity,
    publisher_id: str,
    key_id: str = "test-publisher-key-1",
    attestation_id: str = "attest-test-1",
    seed: bytes = _DEFAULT_SEED,
) -> AgentPackageAttestationQualificationEvidence:
    private_key, public_key_bytes = build_test_attestation_keypair(seed=seed)
    signature_b64 = sign_package_attestation_statement(
        package_identity=package_identity,
        publisher_id=publisher_id,
        key_id=key_id,
        private_key=private_key,
    )
    verifier = build_test_attestation_verifier(
        keys={(publisher_id.strip(), key_id.strip()): public_key_bytes}
    )
    return verifier.verify_qualification_evidence(
        AgentPackageAttestationVerificationRequest(
            package_identity=package_identity,
            publisher_id=publisher_id,
            attestation_id=attestation_id,
            key_id=key_id,
            algorithm=AgentPackageAttestationAlgorithm.ED25519,
            signature_b64=signature_b64,
        )
    )
