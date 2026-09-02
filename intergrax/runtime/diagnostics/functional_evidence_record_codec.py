# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Versioned persistence encoding for ``PlatformFunctionalEvidence`` (DIAG-DURABILITY-D1)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from intergrax.runtime.diagnostics.functional_evidence import (
    PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA,
    PlatformFunctionalEvidence,
)

_PERSISTENCE_SCHEMA = "intergrax.functional_evidence.persistence.v1"
_PAYLOAD_FIELD = "payload"


def encode_functional_evidence_record(
    evidence: PlatformFunctionalEvidence,
) -> dict[str, Any]:
    """Serialize functional evidence for document storage."""
    return {
        "schema_version": _PERSISTENCE_SCHEMA,
        _PAYLOAD_FIELD: evidence.model_dump(mode="json"),
    }


def _normalize_persistence_payload(payload: dict[str, object]) -> dict[str, object]:
    """Restore tuple semantics lost during JSON serialization."""
    normalized = dict(payload)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        prov = dict(provenance)
        upstream = prov.get("upstream_evidence_ids")
        if isinstance(upstream, list):
            prov["upstream_evidence_ids"] = tuple(upstream)
        normalized["provenance"] = prov
    return normalized


def decode_functional_evidence_record(data: object) -> PlatformFunctionalEvidence:
    """Reconstruct typed functional evidence from stored representation."""
    if not isinstance(data, dict):
        raise ValueError("invalid functional evidence persistence record")
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise ValueError("unsupported functional evidence persistence schema")
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ValueError("invalid functional evidence persistence payload")
    if payload.get("schema_version") != PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA:
        raise ValueError("invalid platform functional evidence schema in payload")
    try:
        return PlatformFunctionalEvidence.model_validate(
            _normalize_persistence_payload(payload),
        )
    except ValidationError as exc:
        raise ValueError("invalid functional evidence persistence payload") from exc


def encode_functional_evidence_record_bytes(
    evidence: PlatformFunctionalEvidence,
) -> bytes:
    return json.dumps(
        encode_functional_evidence_record(evidence),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_functional_evidence_record_bytes(raw: bytes) -> PlatformFunctionalEvidence:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid functional evidence persistence bytes") from exc
    return decode_functional_evidence_record(parsed)


__all__ = [
    "decode_functional_evidence_record",
    "decode_functional_evidence_record_bytes",
    "encode_functional_evidence_record",
    "encode_functional_evidence_record_bytes",
]
