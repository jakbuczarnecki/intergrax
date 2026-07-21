# © Artur Czarnecki. All rights reserved.

"""Offline verifier for execution-evidence ProofReceipt — never authorizes."""

from __future__ import annotations

import base64
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from intergrax.contracts.execution_evidence.attestation import HostKeyResolver
from intergrax.contracts.execution_evidence.boundary_event import (
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
)
from intergrax.contracts.execution_evidence.receipt import (
    SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1,
    ProofReceipt,
)
from intergrax.contracts.execution_evidence.verification import VerificationResult
from intergrax.contracts.runtime_policy import PolicyAction
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


# PC-10: HostKeyResolver is the stable port name; VerificationKeyResolver remains.
HostKeyResolverPort = HostKeyResolver


class StaticKeyResolver:
    """Local/test ``HostKeyResolver`` — not a production KMS."""

    def __init__(
        self,
        keys: dict[str, bytes],
        *,
        current_key_id: str | None = None,
        deprecated_key_ids: frozenset[str] | None = None,
        allowed_algorithms: frozenset[str] | None = None,
    ) -> None:
        self._keys = dict(keys)
        self._current_key_id = current_key_id or (next(iter(keys), "") if keys else "")
        self._deprecated = set(deprecated_key_ids or ())
        self._allowed_algorithms = set(allowed_algorithms or {ALGORITHM_ED25519})

    def resolve_public_key(self, key_id: str) -> bytes | None:
        return self._keys.get(key_id)

    def current_signing_key_id(self) -> str:
        return self._current_key_id

    def is_algorithm_allowed(self, algorithm: str) -> bool:
        return algorithm in self._allowed_algorithms

    def is_key_deprecated_for_verification(self, key_id: str) -> bool:
        return key_id in self._deprecated


def _verify_policy_bundle_artifact(receipt: ProofReceipt, errors: list[str]) -> None:
    """PC-2: recompute bundle digest and bind decision to pack body."""
    artifact = receipt.policy_bundle_artifact
    if artifact is None:
        errors.append("policy_bundle_artifact_missing")
        return
    event = receipt.execution_boundary_event
    recomputed = artifact.compute_digest()
    if artifact.canonical_digest and artifact.canonical_digest != recomputed:
        errors.append("policy_bundle_canonical_digest_mismatch")
    if event.policy.bundle_digest != recomputed:
        errors.append("policy_bundle_digest_mismatch")
    if event.policy.bundle_id != artifact.bundle_id:
        errors.append("policy_bundle_id_mismatch")
    if event.policy.bundle_version != artifact.version:
        errors.append("policy_bundle_version_mismatch")
    if not event.policy.rule_id.strip():
        errors.append("policy_rule_id_missing")
        return
    matched = next(
        (r for r in artifact.rules if r.rule_id == event.policy.rule_id),
        None,
    )
    if matched is None:
        errors.append("policy_rule_absent_from_bundle")
        return
    if matched.effect.strip():
        try:
            expected = PolicyAction(matched.effect.strip().lower())
        except ValueError:
            errors.append("policy_rule_effect_invalid")
            return
        if event.policy.action is not expected:
            errors.append("policy_action_mismatch_with_rule")


def verify_proof_receipt(
    receipt: ProofReceipt,
    *,
    key_resolver: VerificationKeyResolver | HostKeyResolver,
    require_policy_bundle_artifact: bool = False,
) -> VerificationResult:
    """Recalculate canonical bytes/digest and verify host signature.

    When ``require_policy_bundle_artifact`` is True (or the receipt embeds an
    artifact), also recompute the immutable pack digest and bind the decision
    rule/action to the pack body (PC-2).
    """
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
    if hasattr(key_resolver, "is_algorithm_allowed"):
        if not key_resolver.is_algorithm_allowed(receipt.host_attestation.algorithm):  # type: ignore[attr-defined]
            schema_valid = False
            errors.append("algorithm_not_allowed")

    payload = canonical_json_bytes(event.canonical_payload())
    computed_digest = stable_payload_hash_from_bytes(payload)
    if computed_digest == receipt.host_attestation.payload_digest:
        digest_valid = True
    else:
        errors.append("digest_mismatch")

    if require_policy_bundle_artifact or receipt.policy_bundle_artifact is not None:
        _verify_policy_bundle_artifact(receipt, errors)

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
