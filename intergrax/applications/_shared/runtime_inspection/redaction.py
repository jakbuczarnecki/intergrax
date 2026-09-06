# © Artur Czarnecki. All rights reserved.

"""Inspection output redaction (P1.4)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def _redact_mapping(path: str, value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _redact_mapping(f"{path}.{key}" if path else key, item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_redact_mapping(f"{path}[{index}]", item) for index, item in enumerate(value)]
    encoded = encode_provenance_value(path, value)
    if encoded is None:
        return None
    return encoded


def redacted_profile_snapshot(profile: ApplicationEnvironmentProfile) -> dict[str, Any]:
    """Serialize profile for inspection without leaking secret-like values."""
    payload = profile.model_dump(mode="json")
    redacted = _redact_mapping("", payload)
    assert isinstance(redacted, dict)
    return redacted


def profile_contains_no_raw_secrets(
    payload: dict[str, Any] | str,
    *,
    raw_secret: str,
) -> bool:
    """Return True when raw secret value is absent from serialized inspection output."""
    serialized = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else payload
    return raw_secret not in serialized
