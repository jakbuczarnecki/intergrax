# © Artur Czarnecki. All rights reserved.

"""Output PII redaction middleware (IDEAL-23.6)."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s\-()]{7,}\d\b")


def redact_pii(text: str) -> str:
    """Redact common email and phone patterns from model output."""
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
