# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from pydantic import BaseModel, Field

# Platform content digest convention: ``sha256:<64 lowercase hex>``.
_CONTENT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_content_digest(value: str) -> str:
    """Validate a content digest string using the platform ``sha256:`` convention."""
    normalized = value.strip()
    if not _CONTENT_DIGEST_RE.match(normalized):
        raise ValueError("digest must match sha256:<64 lowercase hex>")
    return normalized


def compute_sha256_content_digest(canonical_bytes: bytes) -> str:
    """Hash canonical bytes into the platform ``sha256:`` digest format."""
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return f"sha256:{digest}"


class ValidationResult(BaseModel):
    """Result of validating agent output (canonical architecture §13, §29)."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
