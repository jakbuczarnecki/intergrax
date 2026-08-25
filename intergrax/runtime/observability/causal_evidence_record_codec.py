# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned persistence encoding for ``PlatformCausalEvidence`` (DIAG-1D)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    PlatformCausalEvidence,
)

_PERSISTENCE_SCHEMA = "intergrax.causal_evidence.persistence.v1"
_PAYLOAD_FIELD = "payload"


def encode_causal_evidence_record(evidence: PlatformCausalEvidence) -> dict[str, Any]:
    """Serialize evidence for document/KV storage."""
    return {
        "schema_version": _PERSISTENCE_SCHEMA,
        _PAYLOAD_FIELD: evidence.model_dump(mode="json"),
    }


def decode_causal_evidence_record(data: object) -> PlatformCausalEvidence:
    """Reconstruct typed evidence from stored representation."""
    if not isinstance(data, dict):
        raise ValueError("invalid causal evidence persistence record")
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise ValueError("unsupported causal evidence persistence schema")
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ValueError("invalid causal evidence persistence payload")
    if payload.get("schema_version") != PLATFORM_CAUSAL_EVIDENCE_SCHEMA:
        raise ValueError("invalid platform causal evidence schema in payload")
    try:
        return PlatformCausalEvidence.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid causal evidence persistence payload") from exc


def encode_causal_evidence_record_bytes(evidence: PlatformCausalEvidence) -> bytes:
    return json.dumps(
        encode_causal_evidence_record(evidence),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_causal_evidence_record_bytes(raw: bytes) -> PlatformCausalEvidence:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid causal evidence persistence bytes") from exc
    return decode_causal_evidence_record(parsed)
