# © Artur Czarnecki. All rights reserved.

"""EBE-9 host-side attestation: canonical statement signing for boundary events."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.attestation.canonical_json import (
    canonical_json_bytes,
    stable_payload_hash,
)
from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1
from intergrax.utils.time_provider import SystemTimeProvider

HOST_ATTESTATION_CONTEXT = "boundaryattest.host-attestation.v1"
HOST_ATTESTATION_ENVELOPE_SCHEMA = "host_attestation_envelope.v1"
BOUNDARY_EVENT_SCHEMA_ID = "execution_boundary_event.v1"
SIGNATURE_ALGORITHM = "Ed25519"

# PoC/demo seed — pinned in partner_handoff/ebe9_golden_vector.v1.json (dev fallback only).
POC_ATTESTATION_DEMO_SIGNING_SEED = bytes.fromhex(
    "7d4b6e2f9a1c8053e6f2a4d8b0c7e9153f8a2d6c4b9e0175a3d8f2c6b4e190ab"
)


class HostAttestationEnvelopeV1(BaseModel):
    """Signed host/runtime claim over one execution boundary event."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["host_attestation_envelope.v1"] = HOST_ATTESTATION_ENVELOPE_SCHEMA
    context: Literal["boundaryattest.host-attestation.v1"] = HOST_ATTESTATION_CONTEXT
    payload_schema_id: Literal["execution_boundary_event.v1"] = BOUNDARY_EVENT_SCHEMA_ID
    signed_payload_hash: str
    signature_algorithm: Literal["Ed25519"] = SIGNATURE_ALGORITHM
    public_key_id: str
    signed_at: str
    signature: str = Field(
        description="Base64-encoded Ed25519 signature over canonical host-attestation statement bytes.",
    )


@dataclass(frozen=True, slots=True)
class HostAttestationSealer:
    """Sign canonical host-attestation statements with a pinned Ed25519 key."""

    public_key_id: str
    _private_key: Ed25519PrivateKey

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes_raw()

    @property
    def public_key_base64(self) -> str:
        return base64.b64encode(self.public_key_bytes).decode("ascii")

    def build_statement(
        self,
        *,
        signed_payload_hash: str,
        signed_at: str,
    ) -> dict[str, str]:
        return {
            "context": HOST_ATTESTATION_CONTEXT,
            "payload_schema_id": BOUNDARY_EVENT_SCHEMA_ID,
            "signed_payload_hash": signed_payload_hash,
            "signature_algorithm": SIGNATURE_ALGORITHM,
            "public_key_id": self.public_key_id,
            "signed_at": signed_at,
        }

    def sign_statement(self, statement: dict[str, str]) -> bytes:
        return self._private_key.sign(canonical_json_bytes(statement))

    def seal_event(
        self,
        event: ExecutionBoundaryEventV1,
        *,
        signed_at: str | None = None,
    ) -> tuple[ExecutionBoundaryEventV1, HostAttestationEnvelopeV1]:
        unsigned_payload = event.model_copy(update={"signed": False}).model_dump(mode="json")
        payload_hash = stable_payload_hash(unsigned_payload)
        resolved_signed_at = signed_at or SystemTimeProvider.utc_now().isoformat()
        statement = self.build_statement(
            signed_payload_hash=payload_hash,
            signed_at=resolved_signed_at,
        )
        signature_bytes = self.sign_statement(statement)
        envelope = HostAttestationEnvelopeV1(
            signed_payload_hash=payload_hash,
            public_key_id=self.public_key_id,
            signed_at=resolved_signed_at,
            signature=base64.b64encode(signature_bytes).decode("ascii"),
        )
        signed_event = event.model_copy(update={"signed": True})
        return signed_event, envelope


def unsigned_event_payload(event: ExecutionBoundaryEventV1) -> dict[str, Any]:
    return event.model_copy(update={"signed": False}).model_dump(mode="json")


def statement_from_envelope(envelope: HostAttestationEnvelopeV1) -> dict[str, str]:
    return {
        "context": envelope.context,
        "payload_schema_id": envelope.payload_schema_id,
        "signed_payload_hash": envelope.signed_payload_hash,
        "signature_algorithm": envelope.signature_algorithm,
        "public_key_id": envelope.public_key_id,
        "signed_at": envelope.signed_at,
    }


def verify_host_attestation(
    event: ExecutionBoundaryEventV1 | dict[str, Any],
    envelope: HostAttestationEnvelopeV1 | dict[str, Any],
    *,
    public_key: Ed25519PublicKey | bytes,
) -> None:
    """Verify event digest and Ed25519 statement signature (raises on failure)."""
    if isinstance(event, ExecutionBoundaryEventV1):
        event_payload = unsigned_event_payload(event)
    else:
        event_payload = {key: value for key, value in event.items() if key != "host_attestation"}
        event_payload["signed"] = False
    if isinstance(envelope, dict):
        envelope = HostAttestationEnvelopeV1.model_validate(envelope)

    computed_hash = stable_payload_hash(event_payload)
    if computed_hash != envelope.signed_payload_hash:
        raise ValueError("signed_payload_hash does not match canonical event digest")

    statement = statement_from_envelope(envelope)
    if statement["public_key_id"] != envelope.public_key_id:
        raise ValueError("public_key_id mismatch in host-attestation statement")

    signature = base64.b64decode(envelope.signature.encode("ascii"))
    key = (
        public_key
        if isinstance(public_key, Ed25519PublicKey)
        else Ed25519PublicKey.from_public_bytes(public_key)
    )
    key.verify(signature, canonical_json_bytes(statement))


def _decode_signing_key_material(raw: str) -> bytes:
    text = raw.strip()
    if not text:
        raise ValueError("empty signing key material")
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        decoded = bytes.fromhex(text)
    if len(decoded) != 32:
        raise ValueError("Ed25519 signing key must be 32 bytes (seed or raw private key)")
    return decoded


def build_host_attestation_sealer(
    *,
    public_key_id: str,
    private_key_material: bytes | str | None,
) -> HostAttestationSealer | None:
    if not public_key_id.strip():
        return None
    if private_key_material is None:
        return None
    seed = (
        _decode_signing_key_material(private_key_material)
        if isinstance(private_key_material, str)
        else private_key_material
    )
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    return HostAttestationSealer(public_key_id=public_key_id.strip(), _private_key=private_key)


def resolve_host_attestation_sealer_from_env(
    *,
    enabled: bool,
    public_key_id: str,
) -> HostAttestationSealer | None:
    if not enabled:
        return None
    raw_key = (os.getenv("INTERGRAX_EBE_HOST_SIGNING_KEY") or "").strip()
    if raw_key:
        return build_host_attestation_sealer(
            public_key_id=public_key_id,
            private_key_material=raw_key,
        )
    # PoC/lab default — documented in partner_handoff/ebe9_golden_vector.v1.json
    return build_host_attestation_sealer(
        public_key_id=public_key_id,
        private_key_material=POC_ATTESTATION_DEMO_SIGNING_SEED,
    )
