# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Safe summaries for canonical runtime inspection output (INSPECT-01-A)."""

from __future__ import annotations

import re

_EMBEDDED_SECRET_MARKERS: frozenset[str] = frozenset(
    {
        "secret",
        "token",
        "api_key",
        "credential",
        "password",
        "authorization",
    },
)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|credential|password|authorization)\s*[:=]\s*\S+",
)
_REDACTED = "[redacted]"


def sanitize_inspection_text(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    if any(marker in lowered for marker in _EMBEDDED_SECRET_MARKERS):
        return _REDACTED
    if _KEY_VALUE_SECRET_PATTERN.search(text):
        return _REDACTED
    return text


def payload_contains_raw_secret(payload: str, *, raw_secret: str) -> bool:
    return raw_secret in payload


__all__ = ["payload_contains_raw_secret", "sanitize_inspection_text"]
