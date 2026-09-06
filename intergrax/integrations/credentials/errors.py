# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe credential-resolution error surfaces (P1.7)."""

from __future__ import annotations

import re

_SECRET_LIKE_FRAGMENT = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|credential|private[_-]?key)\s*[:=]\s*\S+",
)


def sanitize_credential_error_message(message: str) -> str:
    """Remove secret-like fragments from externally visible error text."""
    cleaned = _SECRET_LIKE_FRAGMENT.sub("<redacted>", message)
    return cleaned.strip() or "credential resolution failed"
