# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic JSON digest for policy bundles and attestation payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Stable UTF-8 JSON bytes for cross-runtime digest agreement."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def canonical_json_text(payload: dict[str, Any]) -> str:
    return canonical_json_bytes(payload).decode("utf-8")


def stable_payload_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"
