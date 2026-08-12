# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic content-addressed digests for Agent Distribution contracts."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from intergrax.runtime.attestation.canonical_json import stable_payload_hash

_PACKAGE_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def normalize_package_digest(value: str) -> str:
    """Normalize and validate a canonical agent artifact package digest."""
    normalized = value.strip().lower()
    if not _PACKAGE_DIGEST_RE.match(normalized):
        raise ValueError("package_digest must be sha256:<64 lowercase hex>")
    return normalized


def normalize_optional_package_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_package_digest(value)


def content_digest_for_model(model: BaseModel) -> str:
    """Return a stable sha256 digest for a frozen distribution contract model."""
    payload: dict[str, Any] = model.model_dump(mode="json")
    return stable_payload_hash(payload)
