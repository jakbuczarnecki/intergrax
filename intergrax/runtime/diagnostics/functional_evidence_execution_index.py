# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Derived execution index contract for functional evidence queries (DIAG-FUNCTIONAL-READ-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)

_INDEX_SCHEMA_V1 = "intergrax.functional_evidence.index.v1"
_INDEX_SCHEMA_V2 = "intergrax.functional_evidence.index.v2"
_EXEC_ROW_PREFIX = "exec:"
_EXECIDX_ROW_PREFIX = "execidx:"
_EVIDENCE_ID_FIELD = "evidence_id"
_RECORDED_AT_FIELD = "recorded_at"
_KIND_FIELD = "kind"
_ATTEMPT_ID_FIELD = "attempt_id"
_MAX_INDEX_MICROS = 10**16
_MIN_INDEX_MICROS = 0
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class FunctionalEvidenceIndexTimestampError(ValueError):
    """Raised when ``recorded_at`` is outside the supported execution-index timestamp range."""


@dataclass(frozen=True, slots=True)
class DecodedExecutionIndexV1:
    evidence_id: str


@dataclass(frozen=True, slots=True)
class DecodedExecutionIndexV2:
    evidence_id: str
    recorded_at: datetime
    kind: PipelineEvidenceKind
    attempt_id: AttemptId | None
    schema_version: Literal["intergrax.functional_evidence.index.v2"]


def execution_index_v1_row_key(*, task_id: TaskId, run_id: RunId, evidence_id: str) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:{evidence_id}"


def execution_index_v1_row_key_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"


def execution_index_v2_row_key(
    *,
    task_id: TaskId,
    run_id: RunId,
    recorded_at: datetime,
    evidence_id: str,
) -> str:
    micros = _datetime_to_epoch_micros(recorded_at)
    return f"{_EXECIDX_ROW_PREFIX}{task_id}:{run_id}:{micros:020d}:{evidence_id}"


def execution_index_v2_row_key_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXECIDX_ROW_PREFIX}{task_id}:{run_id}:"


def execution_index_v2_row_key_from_evidence(evidence: PlatformFunctionalEvidence) -> str:
    return execution_index_v2_row_key(
        task_id=evidence.scope.task_id,
        run_id=evidence.scope.run_id,
        recorded_at=evidence.provenance.recorded_at,
        evidence_id=str(evidence.evidence_id),
    )


def encode_execution_index_v1(evidence_id: str) -> dict[str, str]:
    return {
        "schema_version": _INDEX_SCHEMA_V1,
        _EVIDENCE_ID_FIELD: evidence_id,
    }


def encode_execution_index_v2(evidence: PlatformFunctionalEvidence) -> dict[str, str]:
    recorded_at = evidence.provenance.recorded_at
    _validate_index_timestamp(recorded_at)
    payload: dict[str, str] = {
        "schema_version": _INDEX_SCHEMA_V2,
        _EVIDENCE_ID_FIELD: str(evidence.evidence_id),
        _RECORDED_AT_FIELD: _encode_datetime(recorded_at),
        _KIND_FIELD: evidence.kind.value,
    }
    if evidence.scope.attempt_id is not None:
        payload[_ATTEMPT_ID_FIELD] = str(evidence.scope.attempt_id)
    return payload


def decode_execution_index_v1(data: object) -> DecodedExecutionIndexV1:
    if not isinstance(data, dict):
        raise ValueError("invalid functional evidence execution index")
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA_V1:
        raise ValueError("unsupported functional evidence execution index schema")
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("invalid functional evidence execution index reference")
    return DecodedExecutionIndexV1(evidence_id=evidence_id)


def decode_execution_index_v2(data: object) -> DecodedExecutionIndexV2:
    if not isinstance(data, dict):
        raise ValueError("invalid functional evidence execution index")
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA_V2:
        raise ValueError("unsupported functional evidence execution index schema")
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    recorded_at = data.get(_RECORDED_AT_FIELD)
    kind = data.get(_KIND_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("invalid functional evidence execution index reference")
    if not isinstance(recorded_at, str) or not recorded_at:
        raise ValueError("invalid functional evidence execution index timestamp")
    if not isinstance(kind, str) or kind not in {item.value for item in PipelineEvidenceKind}:
        raise ValueError("invalid functional evidence execution index kind")
    parsed_recorded_at = _decode_datetime(recorded_at)
    _validate_index_timestamp(parsed_recorded_at)
    attempt_id: AttemptId | None = None
    raw_attempt_id = data.get(_ATTEMPT_ID_FIELD)
    if raw_attempt_id is not None:
        if not isinstance(raw_attempt_id, str) or not raw_attempt_id:
            raise ValueError("invalid functional evidence execution index attempt_id")
        attempt_id = AttemptId(raw_attempt_id)
    return DecodedExecutionIndexV2(
        evidence_id=evidence_id,
        recorded_at=parsed_recorded_at,
        kind=PipelineEvidenceKind(kind),
        attempt_id=attempt_id,
        schema_version=_INDEX_SCHEMA_V2,
    )


def index_v2_matches_filters(
    indexed: DecodedExecutionIndexV2,
    *,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> bool:
    if kind is not None and indexed.kind is not kind:
        return False
    if attempt_id is not None and indexed.attempt_id != attempt_id:
        return False
    return True


def _datetime_to_epoch_micros(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = value.astimezone(UTC) - _UTC_EPOCH
    return (
        delta.days * 86_400 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _validate_index_timestamp(value: datetime) -> None:
    micros = _datetime_to_epoch_micros(value)
    if micros < _MIN_INDEX_MICROS or micros > _MAX_INDEX_MICROS:
        raise FunctionalEvidenceIndexTimestampError(
            "functional evidence execution index timestamp out of supported range",
        )


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _decode_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


__all__ = [
    "DecodedExecutionIndexV1",
    "DecodedExecutionIndexV2",
    "FunctionalEvidenceIndexTimestampError",
    "decode_execution_index_v1",
    "decode_execution_index_v2",
    "encode_execution_index_v1",
    "encode_execution_index_v2",
    "execution_index_v1_row_key",
    "execution_index_v1_row_key_prefix",
    "execution_index_v2_row_key",
    "execution_index_v2_row_key_from_evidence",
    "execution_index_v2_row_key_prefix",
    "index_v2_matches_filters",
]
