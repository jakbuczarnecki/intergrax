# © Artur Czarnecki. All rights reserved.

"""Ed25519 HostAttestor — local/test signer; KMS/HSM-replaceable via DI."""

from __future__ import annotations

import base64
import hashlib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from intergrax.contracts.execution_evidence.attestation import HostAttestation
from intergrax.core.security.security_configuration import (
    DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
    SecurityConfigurationError,
)

ALGORITHM_ED25519 = "Ed25519"


class Ed25519HostAttestor:
    """Host-layer signer over canonical payload bytes.

    Signature is Ed25519 over the raw canonical payload bytes (not a separate
    statement envelope). ``payload_digest`` is ``sha256:<hex>`` of those bytes.
    Production hosts may replace this implementation with a KMS/HSM-backed
    ``HostAttestor`` — no production key custody is claimed here.
    """

    def __init__(
        self,
        *,
        key_id: str,
        private_key: Ed25519PrivateKey,
        clock: Callable[[], datetime] | None = None,
        attestation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not key_id.strip():
            raise ValueError("key_id must be non-empty")
        self._key_id = key_id.strip()
        self._private_key = private_key
        self._clock = clock or (lambda: datetime.now(UTC))
        self._attestation_id_factory = attestation_id_factory or (lambda: f"att-{uuid.uuid4().hex}")

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes_raw()

    @property
    def public_key_base64(self) -> str:
        return base64.b64encode(self.public_key_bytes).decode("ascii")

    def attest(self, payload: bytes, *, schema: str) -> HostAttestation:
        if not schema.strip():
            raise ValueError("schema must be non-empty")
        if not payload:
            raise ValueError("payload must be non-empty")
        digest = stable_payload_hash_from_bytes(payload)
        signature = self._private_key.sign(payload)
        return HostAttestation(
            attestation_id=self._attestation_id_factory(),
            algorithm=ALGORITHM_ED25519,
            key_id=self._key_id,
            payload_digest=digest,
            signature=base64.b64encode(signature).decode("ascii"),
            signed_at=self._clock(),
            payload_schema=schema.strip(),
        )


def stable_payload_hash_from_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def resolve_demo_attestation_seed(*, strict_profile: bool) -> bytes:
    """Return the public PoC demo seed for lab and proof paths only (PDC-D6-R027).

    The seed is imported lazily so that no product module holds it at import time. A strict
    (product) profile can never resolve it: that raises a typed configuration error.
    """
    if strict_profile:
        raise SecurityConfigurationError(
            item="demo_attestation_seed",
            reason_code=DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
            message="the demo attestation signing seed is not available in a strict profile",
        )
    from intergrax.runtime.attestation.host_attestation import POC_ATTESTATION_DEMO_SIGNING_SEED

    return POC_ATTESTATION_DEMO_SIGNING_SEED


def build_deterministic_test_attestor(
    *,
    key_id: str = "governed-contractor-test-host-1",
    seed: bytes | None = None,
    clock: Callable[[], datetime] | None = None,
    attestation_id: str = "att-test-fixed",
    strict_profile: bool = False,
) -> Ed25519HostAttestor:
    """Deterministic local attestor for offline lab and proof paths (PoC seed, not custody).

    Calling this function is the named lab/proof opt-in. Pass ``strict_profile=True`` from any
    path that runs under a strict profile: the default demo seed is then refused.
    """
    private_key = Ed25519PrivateKey.from_private_bytes(
        seed if seed is not None else resolve_demo_attestation_seed(strict_profile=strict_profile)
    )
    return Ed25519HostAttestor(
        key_id=key_id,
        private_key=private_key,
        clock=clock,
        attestation_id_factory=lambda: attestation_id,
    )


def public_key_from_attestor(attestor: Ed25519HostAttestor) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(attestor.public_key_bytes)
