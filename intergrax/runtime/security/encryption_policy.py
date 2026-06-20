# © Artur Czarnecki. All rights reserved.

"""Encryption enforcement for RESTRICTED data classification (Phase ENC-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.data_classification import DataClassification


@dataclass(frozen=True, slots=True)
class EncryptionEnforcementDecision:
    """Whether a payload may proceed under encryption policy."""

    allowed: bool
    reasons: tuple[str, ...] = ()


def _classification_from_payload(payload: dict[str, object]) -> DataClassification | None:
    raw = payload.get("data_classification") or payload.get("classification")
    if raw is None:
        value = payload.get("value")
        if isinstance(value, dict):
            raw = value.get("data_classification") or value.get("classification")
    if raw is None:
        return None
    try:
        return DataClassification(str(raw).lower())
    except ValueError:
        return None


def evaluate_encryption_enforcement(
    *,
    payload: dict[str, object],
    secrets_store_configured: bool,
    enforcement_enabled: bool,
) -> EncryptionEnforcementDecision:
    """Fail-closed when RESTRICTED/CONFIDENTIAL data lacks secrets backend on strict hosts."""
    if not enforcement_enabled:
        return EncryptionEnforcementDecision(allowed=True)
    classification = _classification_from_payload(payload)
    if classification is None:
        return EncryptionEnforcementDecision(allowed=True)
    if not classification.requires_encryption():
        return EncryptionEnforcementDecision(allowed=True)
    if secrets_store_configured:
        return EncryptionEnforcementDecision(allowed=True)
    return EncryptionEnforcementDecision(
        allowed=False,
        reasons=(
            f"{classification.value} data requires configured secrets_store integration",
        ),
    )
