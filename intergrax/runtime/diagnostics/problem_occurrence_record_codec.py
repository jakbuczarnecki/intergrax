# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Versioned persistence encoding for durable ProblemOccurrence rows (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrence
from intergrax.runtime.diagnostics.problem_occurrence_id import (
    ProblemOccurrenceId,
    problem_occurrence_id_for,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_record_codec import (
    decode_problem_occurrence_payload,
    encode_problem_occurrence_payload,
)

_PERSISTENCE_SCHEMA = "intergrax.diagnostic_problem_occurrence.persistence.v1"
_PAYLOAD_FIELD = "payload"
_OCCURRENCE_ID_FIELD = "occurrence_id"


def encode_problem_occurrence_record(occurrence: ProblemOccurrence) -> dict[str, object]:
    occurrence_id = problem_occurrence_id_for(occurrence)
    payload = encode_problem_occurrence_payload(occurrence)
    payload[_OCCURRENCE_ID_FIELD] = str(occurrence_id)
    return {
        "schema_version": _PERSISTENCE_SCHEMA,
        _PAYLOAD_FIELD: payload,
    }


def decode_problem_occurrence_record(data: object) -> ProblemOccurrence:
    if not isinstance(data, dict):
        raise ProblemOccurrencePersistenceIntegrityError(
            "invalid diagnostic problem occurrence persistence record",
        )
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise ProblemOccurrencePersistenceIntegrityError(
            "unsupported diagnostic problem occurrence persistence schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ProblemOccurrencePersistenceIntegrityError(
            "invalid diagnostic problem occurrence persistence payload",
        )
    try:
        occurrence = decode_problem_occurrence_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProblemOccurrencePersistenceIntegrityError(
            "malformed diagnostic problem occurrence persistence payload",
        ) from exc
    stored_id = payload.get(_OCCURRENCE_ID_FIELD)
    if not isinstance(stored_id, str) or not stored_id:
        raise ProblemOccurrencePersistenceIntegrityError(
            "malformed diagnostic problem occurrence id",
        )
    if stored_id != str(problem_occurrence_id_for(occurrence)):
        raise ProblemOccurrencePersistenceIntegrityError(
            "diagnostic problem occurrence id does not match payload",
        )
    return occurrence


def encode_occurrence_stats_record(
    *,
    occurrence_count: int,
    first_seen_at: datetime,
    last_seen_at: datetime,
) -> dict[str, object]:
    if occurrence_count < 1:
        raise ValueError("occurrence_count must be positive for stats record")
    return {
        "schema_version": _PERSISTENCE_SCHEMA,
        _PAYLOAD_FIELD: {
            "occurrence_count": occurrence_count,
            "first_seen_at": _encode_datetime(first_seen_at),
            "last_seen_at": _encode_datetime(last_seen_at),
        },
    }


def decode_occurrence_stats_record(data: object) -> tuple[int, datetime, datetime]:
    if not isinstance(data, dict):
        raise ProblemOccurrencePersistenceIntegrityError(
            "invalid diagnostic problem occurrence stats record",
        )
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise ProblemOccurrencePersistenceIntegrityError(
            "unsupported diagnostic problem occurrence stats schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ProblemOccurrencePersistenceIntegrityError(
            "invalid diagnostic problem occurrence stats payload",
        )
    try:
        count = int(payload["occurrence_count"])  # type: ignore[arg-type]
        first_seen_at = _decode_datetime(payload["first_seen_at"])
        last_seen_at = _decode_datetime(payload["last_seen_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProblemOccurrencePersistenceIntegrityError(
            "malformed diagnostic problem occurrence stats payload",
        ) from exc
    if count < 1:
        raise ProblemOccurrencePersistenceIntegrityError(
            "invalid diagnostic problem occurrence stats count",
        )
    return count, first_seen_at, last_seen_at


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return value.isoformat()


def _decode_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("datetime must be ISO string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return parsed
