# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Versioned persistence encoding for durable ProblemOccurrence rows (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from datetime import datetime

from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrence
from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
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
