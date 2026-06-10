# © Artur Czarnecki. All rights reserved.

"""Cryptographic signing for critical harness actions (AUDIT-IDEAL-4.1)."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class CriticalActionKind(str, Enum):
    POLICY_OVERRIDE = "policy_override"
    PRODUCTION_PROMOTION = "production_promotion"
    ADAPTIVE_APPLY = "adaptive_apply"
    SECURITY_CONFIG_CHANGE = "security_config_change"


class CriticalActionPayload(BaseModel):
    action_id: str
    action_kind: CriticalActionKind
    tenant_id: str
    actor_id: str
    resource: str
    details: dict[str, str] = Field(default_factory=dict)


class CriticalActionSignature(BaseModel):
    schema_version: str = "1.0.0"
    action_id: str
    action_kind: CriticalActionKind
    payload_digest: str
    signature: str
    signed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _canonical_payload_bytes(payload: CriticalActionPayload) -> bytes:
    return json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode("utf-8")


def digest_critical_action_payload(payload: CriticalActionPayload) -> str:
    return hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()


def sign_critical_action(*, secret: str, payload: CriticalActionPayload) -> CriticalActionSignature:
    """Sign a critical action payload with HMAC-SHA256."""
    if not secret.strip():
        raise ValueError("critical action signing secret must be non-empty")
    digest = digest_critical_action_payload(payload)
    signature = hmac.new(
        secret.encode("utf-8"),
        digest.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return CriticalActionSignature(
        action_id=payload.action_id,
        action_kind=payload.action_kind,
        payload_digest=digest,
        signature=signature,
    )


def verify_critical_action_signature(
    *,
    secret: str,
    payload: CriticalActionPayload,
    signature: CriticalActionSignature,
) -> bool:
    """Verify HMAC signature and payload digest for a critical action."""
    expected = sign_critical_action(secret=secret, payload=payload)
    if expected.payload_digest != signature.payload_digest:
        return False
    return hmac.compare_digest(expected.signature, signature.signature)
