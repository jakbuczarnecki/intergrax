# © Artur Czarnecki. All rights reserved.

"""Safe error normalization and secret redaction for proof evidence."""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|authorization|bearer|token|secret|password)\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)sk-[a-z0-9]{10,}"),
    re.compile(r"(?i)hf_[a-z0-9]{10,}"),
)
_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),
    re.compile(r"/(?:home|Users|var|tmp|data)/[^\s\"']+"),
)
_REDACTED = "[REDACTED]"


def redact_text(value: str) -> str:
    text = value
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    for pattern in _PATH_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def safe_error_excerpt(exc: BaseException, *, limit: int = 240) -> str:
    message = redact_text(f"{exc.__class__.__name__}: {exc}")
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def safe_error_type(exc: BaseException) -> str:
    return exc.__class__.__name__


def assert_no_secret_leak(payload: Any) -> str | None:
    text = redact_text(str(payload))
    if _REDACTED in text and text != str(payload):
        return "proof_secret_leak_detected"
    lowered = str(payload).lower()
    forbidden = (
        "authorization:",
        "bearer ",
        "api_key",
        "hf_",
        "sk-",
        "password=",
    )
    for token in forbidden:
        if token in lowered:
            return "proof_secret_leak_detected"
    return None


def normalize_provider_error(exc: BaseException) -> tuple[str, str]:
    return safe_error_type(exc), safe_error_excerpt(exc)
