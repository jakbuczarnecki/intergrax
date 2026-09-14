# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned persistence encoding for ``PlatformCausalEvidence`` (DIAG-1D)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.contracts.npsc5f_compatibility import (
    ForbiddenPlatformCausalEvidenceV1WriteError,
)
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.platform_causal_evidence_codec import (
    DecodedPlatformCausalEvidence,
    decode_platform_causal_evidence_payload,
    require_complete_v2,
)

_PERSISTENCE_SCHEMA = "intergrax.causal_evidence.persistence.v1"
_PAYLOAD_FIELD = "payload"


def encode_causal_evidence_record(evidence: PlatformCausalEvidence) -> dict[str, Any]:
    """Serialize v2 evidence for document/KV storage (v1 platform write forbidden)."""
    payload = evidence.model_dump(mode="json")
    return {
        "schema_version": _PERSISTENCE_SCHEMA,
        _PAYLOAD_FIELD: payload,
    }


def decode_causal_evidence_record(data: object) -> DecodedPlatformCausalEvidence:
    """Reconstruct typed evidence from stored representation (v1 + v2 read)."""
    if not isinstance(data, dict):
        raise ValueError("invalid causal evidence persistence record")
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise ValueError("unsupported causal evidence persistence schema")
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ValueError("invalid causal evidence persistence payload")
    return decode_platform_causal_evidence_payload(payload)


def decode_causal_evidence_record_v2(data: object) -> PlatformCausalEvidence:
    """Strict v2-only decode for paths that require complete ExecutionId correlation."""
    return require_complete_v2(decode_causal_evidence_record(data))


def encode_causal_evidence_record_bytes(evidence: PlatformCausalEvidence) -> bytes:
    return json.dumps(
        encode_causal_evidence_record(evidence),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_causal_evidence_record_bytes(raw: bytes) -> DecodedPlatformCausalEvidence:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid causal evidence persistence bytes") from exc
    return decode_causal_evidence_record(parsed)


def decode_causal_evidence_record_bytes_v2(raw: bytes) -> PlatformCausalEvidence:
    return require_complete_v2(decode_causal_evidence_record_bytes(raw))


def forbid_platform_causal_evidence_v1_write(evidence_payload: dict[str, Any]) -> None:
    """Guard for alternate persistence encodings that embed platform payloads directly."""
    schema_version = evidence_payload.get("schema_version")
    if schema_version == "platform_causal_evidence.v1":
        raise ForbiddenPlatformCausalEvidenceV1WriteError(
            "platform_causal_evidence.v1 write is retired; use v2 only",
        )
