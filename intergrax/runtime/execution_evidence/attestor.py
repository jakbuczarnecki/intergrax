# © Artur Czarnecki. All rights reserved.

"""Ed25519 HostAttestor — local/test signer; KMS/HSM-replaceable via DI."""

from __future__ import annotations

import base64
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from intergrax.contracts.execution_evidence.attestation import HostAttestation
from intergrax.runtime.attestation.host_attestation import POC_ATTESTATION_DEMO_SIGNING_SEED

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
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._attestation_id_factory = attestation_id_factory or (
            lambda: f"att-{uuid.uuid4().hex}"
        )

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


def build_deterministic_test_attestor(
    *,
    key_id: str = "governed-contractor-test-host-1",
    seed: bytes | None = None,
    clock: Callable[[], datetime] | None = None,
    attestation_id: str = "att-test-fixed",
) -> Ed25519HostAttestor:
    """Deterministic local attestor for offline tests (PoC seed, not custody)."""
    private_key = Ed25519PrivateKey.from_private_bytes(
        seed if seed is not None else POC_ATTESTATION_DEMO_SIGNING_SEED
    )
    return Ed25519HostAttestor(
        key_id=key_id,
        private_key=private_key,
        clock=clock,
        attestation_id_factory=lambda: attestation_id,
    )


def public_key_from_attestor(attestor: Ed25519HostAttestor) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(attestor.public_key_bytes)
