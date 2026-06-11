# © Artur Czarnecki. All rights reserved.

"""PII redaction helpers for traces and policy messages (architecture §40.8 · ACP-PROD-8)."""

from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(api[_-]?key[\"']?\s*[:=]\s*)[\"']?[A-Za-z0-9._-]{8,}", re.IGNORECASE)

_REDACTED = "[REDACTED]"


def redact_pii_text(text: str) -> str:
    if not text:
        return text
    redacted = _EMAIL_RE.sub("[EMAIL]", text)
    redacted = _PHONE_RE.sub("[PHONE]", redacted)
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _API_KEY_RE.sub(r"\1[REDACTED]", redacted)
    return redacted


def redact_mapping_values(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            result[key] = redact_pii_text(value)
        elif isinstance(value, dict):
            result[key] = redact_mapping_values(value)
        else:
            result[key] = value
    return result
