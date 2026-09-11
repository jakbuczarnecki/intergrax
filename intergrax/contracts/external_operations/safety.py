# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security boundaries for external operation admission (R1)."""

from __future__ import annotations

import re
from typing import Iterable

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9._\-]{20,}", re.IGNORECASE),
    re.compile(r"authorization\s*:\s*\S+", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"token\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
)


class ExternalOperationAdmissionDeniedError(RuntimeError):
    """Admission returned DENY."""


class ExternalOperationApprovalRequiredError(RuntimeError):
    """Admission requires human approval before execution."""


class ExternalOperationExecutionForbiddenError(RuntimeError):
    """Attempt to execute without passing admission governance."""


def sanitize_external_operation_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("sanitized text must be non-empty")
    redacted = stripped
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def assert_no_secrets_in_audit_payload(parts: Iterable[str]) -> None:
    for part in parts:
        for pattern in _SECRET_PATTERNS:
            if pattern.search(part):
                raise ValueError("secret-like material forbidden in audit/evidence payload")
