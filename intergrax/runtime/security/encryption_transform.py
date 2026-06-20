# © Artur Czarnecki. All rights reserved.

"""RESTRICTED payload envelope transform via secrets_store bridge (Phase SEC-EVOL-4)."""

from __future__ import annotations

import base64
import json
from typing import Protocol, runtime_checkable

from intergrax.contracts.data_classification import DataClassification
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.security.encryption_policy import _classification_from_payload


@runtime_checkable
class RestrictedPayloadEncryptor(Protocol):
    """Transform RESTRICTED inline material before memory write or tool output."""

    def encrypt_payload(self, payload: dict[str, object], *, run_id: str) -> dict[str, object]: ...


class HarnessEnvelopeEncryptor:
    """
    Tier-0 envelope transform — stores ciphertext marker and secret ref.

    Tier-3 hosts SHOULD replace with a ``SecretsStore``-backed encryptor in production.
    """

    def __init__(self, *, prefix: str = "restricted") -> None:
        self._prefix = prefix

    def encrypt_payload(self, payload: dict[str, object], *, run_id: str) -> dict[str, object]:
        classification = _classification_from_payload(payload)
        if classification is None or not classification.requires_encryption():
            return payload
        updated = dict(payload)
        value = updated.get("value")
        if not isinstance(value, dict):
            return updated
        envelope = dict(value)
        secret_material = _extract_secret_material(envelope)
        if secret_material is None:
            return updated
        ref = f"{self._prefix}/{run_id}/payload"
        ciphertext = base64.b64encode(secret_material.encode("utf-8")).decode("ascii")
        envelope.pop("secret", None)
        envelope["__harness_encrypted_ref"] = ref
        envelope["__harness_ciphertext"] = ciphertext
        envelope["encryption_envelope"] = "harness.v1"
        updated["value"] = envelope
        return updated


def _extract_secret_material(value: dict[str, object]) -> str | None:
    for key in ("secret", "payload", "content", "data"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return None


class SecretsStorePayloadEncryptor:
    """Persist RESTRICTED material via ``SecretsStore.put_secret`` and replace with ref."""

    def __init__(self, store: SecretsStore, *, prefix: str = "restricted") -> None:
        self._store = store
        self._prefix = prefix

    def encrypt_payload(self, payload: dict[str, object], *, run_id: str) -> dict[str, object]:
        classification = _classification_from_payload(payload)
        if classification is None or not classification.requires_encryption():
            return payload
        updated = dict(payload)
        value = updated.get("value")
        if not isinstance(value, dict):
            return updated
        envelope = dict(value)
        secret_material = _extract_secret_material(envelope)
        if secret_material is None:
            return updated
        ref = f"{self._prefix}/{run_id}/payload"
        self._store.put_secret(ref, secret_material)
        envelope.pop("secret", None)
        envelope["__secret_ref"] = ref
        envelope["data_classification"] = DataClassification.RESTRICTED.value
        envelope["encryption_envelope"] = "secrets_store.v1"
        updated["value"] = envelope
        return updated
