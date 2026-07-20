# © Artur Czarnecki. All rights reserved.

"""Offline verifier for execution-evidence ProofReceipt — never authorizes."""

from __future__ import annotations

import base64
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from intergrax.contracts.execution_evidence.boundary_event import (
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
)
from intergrax.contracts.execution_evidence.receipt import (
    SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1,
    ProofReceipt,
)
from intergrax.contracts.execution_evidence.verification import VerificationResult
from intergrax.runtime.attestation.canonical_json import canonical_json_bytes
from intergrax.runtime.execution_evidence.attestor import (
    ALGORITHM_ED25519,
    stable_payload_hash_from_bytes,
)

_SUPPORTED_SCHEMAS = frozenset(
    {
        SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
        SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1,
    }
)


@runtime_checkable
class VerificationKeyResolver(Protocol):
    def resolve_public_key(self, key_id: str) -> bytes | None:
        """Return raw Ed25519 public key bytes for ``key_id``, or None if unknown."""
        ...


class StaticKeyResolver:
    def __init__(self, keys: dict[str, bytes]) -> None:
        self._keys = dict(keys)

    def resolve_public_key(self, key_id: str) -> bytes | None:
        return self._keys.get(key_id)


def verify_proof_receipt(
    receipt: ProofReceipt,
    *,
    key_resolver: VerificationKeyResolver,
) -> VerificationResult:
    """Recalculate canonical bytes/digest and verify host signature."""
    errors: list[str] = []
    schema_valid = True
    digest_valid = False
    signature_valid = False
    key_id = receipt.host_attestation.key_id

    if receipt.schema_id != SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1:
        schema_valid = False
        errors.append("unsupported_receipt_schema")
    event = receipt.execution_boundary_event
    if event.schema_id != SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1:
        schema_valid = False
        errors.append("unsupported_event_schema")
    if receipt.host_attestation.payload_schema != SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1:
        schema_valid = False
        errors.append("unsupported_attestation_payload_schema")
    if receipt.host_attestation.algorithm != ALGORITHM_ED25519:
        schema_valid = False
        errors.append("unsupported_algorithm")

    payload = canonical_json_bytes(event.canonical_payload())
    computed_digest = stable_payload_hash_from_bytes(payload)
    if computed_digest == receipt.host_attestation.payload_digest:
        digest_valid = True
    else:
        errors.append("digest_mismatch")

    public_key_bytes = key_resolver.resolve_public_key(key_id)
    if public_key_bytes is None:
        errors.append("unknown_key_id")
    elif digest_valid and schema_valid:
        try:
            key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            signature = base64.b64decode(
                receipt.host_attestation.signature.encode("ascii"),
                validate=True,
            )
            key.verify(signature, payload)
            signature_valid = True
        except (InvalidSignature, ValueError, TypeError):
            errors.append("signature_invalid")
    elif public_key_bytes is not None and not digest_valid:
        errors.append("signature_skipped_digest_invalid")

    valid = schema_valid and digest_valid and signature_valid and not errors
    return VerificationResult(
        valid=valid,
        schema_valid=schema_valid,
        digest_valid=digest_valid,
        signature_valid=signature_valid,
        key_id=key_id,
        errors=tuple(errors),
    )
