# © Artur Czarnecki. All rights reserved.

"""Canonical enterprise memory record metadata (MEM-ENT-5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.evidence_claims import validate_evidence_reference_id

__all__ = [
    "MemoryRecordGovernance",
    "MemoryRecordLineage",
    "MemoryRecordSourceType",
    "MemoryRecordTrust",
    "MemoryProvenance",
    "MemoryTrustClass",
    "memory_record_source_from_legacy_string",
    "validate_memory_record_invariants",
]


class MemoryRecordSourceType(str, Enum):
    USER_EXPLICIT = "user_explicit"
    SESSION_EXTRACTION = "session_extraction"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    IMPORT = "import"
    UNKNOWN = "unknown"


class MemoryTrustClass(str, Enum):
    UNKNOWN = "unknown"
    USER_EXPLICIT = "user_explicit"
    MODEL_INFERENCE = "model_inference"
    EXTERNAL_SOURCE = "external_source"
    SYSTEM_GENERATED = "system_generated"


def memory_record_source_from_legacy_string(value: str | None) -> MemoryRecordSourceType:
    normalized = (value or "").strip().lower()
    if normalized in {"session_consolidation", "session_extraction", "session_summarizer"}:
        return MemoryRecordSourceType.SESSION_EXTRACTION
    if normalized in {"user", "user_explicit", "manual"}:
        return MemoryRecordSourceType.USER_EXPLICIT
    if normalized in {"tool", "tool_result"}:
        return MemoryRecordSourceType.TOOL_RESULT
    if normalized == "import":
        return MemoryRecordSourceType.IMPORT
    if normalized == "system":
        return MemoryRecordSourceType.SYSTEM
    return MemoryRecordSourceType.UNKNOWN


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    source_type: MemoryRecordSourceType = MemoryRecordSourceType.UNKNOWN
    source_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    strategy_id: str | None = None
    actor_user_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecordTrust:
    trust_class: MemoryTrustClass = MemoryTrustClass.UNKNOWN
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecordGovernance:
    data_classification: DataClassification = DataClassification.INTERNAL


@dataclass(frozen=True, slots=True)
class MemoryRecordLineage:
    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None


def parse_memory_record_timestamp(field_name: str, value: str) -> datetime:
    """Parse a non-empty ISO-8601 timestamp string for memory record validation.

    Supports offset-aware values, fractional seconds, and a trailing ``Z`` (UTC)
    consistent with other memory-layer ISO parsing. Naive strings remain naive
    (no implicit timezone). Raises ``ValueError`` when the value is empty or malformed.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError(f"invalid {field_name}: empty timestamp")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"invalid {field_name}: malformed ISO timestamp") from None


def _validate_temporal_window(valid_from: str | None, valid_until: str | None) -> None:
    if valid_from is None or valid_until is None:
        return
    from_dt = parse_memory_record_timestamp("valid_from", valid_from)
    until_dt = parse_memory_record_timestamp("valid_until", valid_until)
    from_aware = from_dt.tzinfo is not None
    until_aware = until_dt.tzinfo is not None
    if from_aware != until_aware:
        raise ValueError(
            "valid_from and valid_until must both be timezone-aware or both naive"
        )
    if from_dt > until_dt:
        raise ValueError("valid_from must not be after valid_until")


def _validate_confidence(confidence: float | None) -> None:
    if confidence is None:
        return
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite and between 0.0 and 1.0")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")


def _validate_evidence_refs(evidence_refs: tuple[str, ...]) -> None:
    for ref in evidence_refs:
        validate_evidence_reference_id(ref)


def validate_memory_record_invariants(
    *,
    memory_id: str,
    revision: int,
    valid_from: str | None,
    valid_until: str | None,
    trust: MemoryRecordTrust,
    lineage: MemoryRecordLineage,
    evidence_refs: tuple[str, ...],
) -> None:
    if not (memory_id or "").strip():
        raise ValueError("memory_id must be non-empty")
    if revision < 1:
        raise ValueError("revision must be >= 1")
    _validate_temporal_window(valid_from, valid_until)
    _validate_confidence(trust.confidence)
    _validate_evidence_refs(evidence_refs)
    if lineage.supersedes_memory_id and lineage.supersedes_memory_id == memory_id:
        raise ValueError("memory record cannot supersede itself")
    if lineage.superseded_by_memory_id and lineage.superseded_by_memory_id == memory_id:
        raise ValueError("memory record cannot be superseded by itself")

