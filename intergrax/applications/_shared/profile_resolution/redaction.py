# © Artur Czarnecki. All rights reserved.

"""Redacted provenance value encoding (P1.1)."""

from __future__ import annotations

import json
from enum import Enum

from intergrax.applications._shared.environment_snapshot_wiring import stable_digest_hex

_SENSITIVE_PATH_SUFFIXES: frozenset[str] = frozenset(
    {
        "options",
        "api_key",
        "token",
        "secret",
        "credential",
    },
)


def _is_sensitive_path(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in _SENSITIVE_PATH_SUFFIXES)


def encode_provenance_value(path: str, value: object | None) -> str | None:
    """Encode a value for resolution evidence without leaking secrets."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)):
        if _is_sensitive_path(path):
            return f"hash:{stable_digest_hex(value)[:16]}"
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = sorted(str(item) for item in value)
        return json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if _is_sensitive_path(path):
        return f"hash:{stable_digest_hex(payload)[:16]}"
    if len(payload) > 120:
        return f"hash:{stable_digest_hex(payload)[:16]}"
    return payload
