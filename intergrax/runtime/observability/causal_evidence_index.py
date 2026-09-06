# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Derived execution/transport index contract for causal evidence paging (DG-002 R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence

_INDEX_SCHEMA_V1 = "intergrax.causal_evidence.index.v1"
_INDEX_SCHEMA_V2 = "intergrax.causal_evidence.index.v2"
_EXEC_ROW_PREFIX = "exec:"
_TRANSPORT_ROW_PREFIX = "transport:"
_EVIDENCE_ID_FIELD = "evidence_id"
_RECORDED_AT_FIELD = "recorded_at"
_MAX_INDEX_MICROS = 10**16
_MIN_INDEX_MICROS = 0
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SORT_TOKEN_WIDTH = 20


class CausalEvidenceIndexTimestampError(ValueError):
    """Raised when ``recorded_at`` is outside the supported causal index timestamp range."""


@dataclass(frozen=True, slots=True)
class DecodedCausalEvidenceIndexV1:
    evidence_id: str


@dataclass(frozen=True, slots=True)
class DecodedCausalEvidenceIndexV2:
    evidence_id: str
    recorded_at: datetime
    schema_version: Literal["intergrax.causal_evidence.index.v2"]


def execution_index_v1_row_key(*, task_id: TaskId, run_id: RunId, evidence_id: str) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:{evidence_id}"


def execution_index_v1_row_key_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"


def transport_index_v1_row_key(
    *,
    provider: str,
    transport_task_id: str,
    evidence_id: str,
) -> str:
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:{evidence_id}"


def transport_index_v1_row_key_prefix(*, provider: str, transport_task_id: str) -> str:
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:"


def execution_index_v2_row_key(
    *,
    task_id: TaskId,
    run_id: RunId,
    recorded_at: datetime,
    evidence_id: str,
) -> str:
    sort_token = recorded_at_sort_token(recorded_at)
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:{sort_token}:{evidence_id}"


def execution_index_v2_row_key_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"


def transport_index_v2_row_key(
    *,
    provider: str,
    transport_task_id: str,
    recorded_at: datetime,
    evidence_id: str,
) -> str:
    sort_token = recorded_at_sort_token(recorded_at)
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:{sort_token}:{evidence_id}"


def transport_index_v2_row_key_prefix(*, provider: str, transport_task_id: str) -> str:
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:"


def execution_index_v2_row_key_from_evidence(evidence: PlatformCausalEvidence) -> str:
    return execution_index_v2_row_key(
        task_id=evidence.target.task_id,
        run_id=evidence.target.run_id,
        recorded_at=evidence.recorded_at,
        evidence_id=str(evidence.evidence_id),
    )


def transport_index_v2_row_key_from_evidence(evidence: PlatformCausalEvidence) -> str:
    return transport_index_v2_row_key(
        provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
        recorded_at=evidence.recorded_at,
        evidence_id=str(evidence.evidence_id),
    )


def recorded_at_sort_token(recorded_at: datetime) -> str:
    micros = _datetime_to_epoch_micros(recorded_at)
    return f"{micros:0{_SORT_TOKEN_WIDTH}d}"


def encode_causal_evidence_index_v1(evidence_id: str) -> dict[str, str]:
    return {
        "schema_version": _INDEX_SCHEMA_V1,
        _EVIDENCE_ID_FIELD: evidence_id,
    }


def encode_causal_evidence_index_v2(evidence: PlatformCausalEvidence) -> dict[str, str]:
    recorded_at = evidence.recorded_at
    _validate_index_timestamp(recorded_at)
    return {
        "schema_version": _INDEX_SCHEMA_V2,
        _EVIDENCE_ID_FIELD: str(evidence.evidence_id),
        _RECORDED_AT_FIELD: _encode_datetime(recorded_at),
    }


def decode_causal_evidence_index_v1(data: object) -> DecodedCausalEvidenceIndexV1:
    if not isinstance(data, dict):
        raise ValueError("invalid causal evidence index")
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA_V1:
        raise ValueError("unsupported causal evidence index schema")
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("invalid causal evidence index reference")
    return DecodedCausalEvidenceIndexV1(evidence_id=evidence_id)


def decode_causal_evidence_index_v2(data: object) -> DecodedCausalEvidenceIndexV2:
    if not isinstance(data, dict):
        raise ValueError("invalid causal evidence index")
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA_V2:
        raise ValueError("unsupported causal evidence index schema")
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    recorded_at = data.get(_RECORDED_AT_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("invalid causal evidence index reference")
    if not isinstance(recorded_at, str) or not recorded_at:
        raise ValueError("invalid causal evidence index timestamp")
    parsed_recorded_at = _decode_datetime(recorded_at)
    _validate_index_timestamp(parsed_recorded_at)
    return DecodedCausalEvidenceIndexV2(
        evidence_id=evidence_id,
        recorded_at=parsed_recorded_at,
        schema_version=_INDEX_SCHEMA_V2,
    )


def is_v2_index_row_key(row_key: str, *, row_prefix: str) -> bool:
    if not row_key.startswith(row_prefix):
        return False
    suffix = row_key[len(row_prefix) :]
    parts = suffix.split(":")
    if len(parts) != 2:
        return False
    sort_token, _evidence_id = parts
    return len(sort_token) == _SORT_TOKEN_WIDTH and sort_token.isdigit()


def row_key_sort_token(row_key: str, *, row_prefix: str) -> str | None:
    if not is_v2_index_row_key(row_key, row_prefix=row_prefix):
        return None
    suffix = row_key[len(row_prefix) :]
    return suffix.split(":", 1)[0]


def v2_index_matches_row_key(
    indexed: DecodedCausalEvidenceIndexV2,
    *,
    row_key: str,
    row_prefix: str,
) -> bool:
    token = row_key_sort_token(row_key, row_prefix=row_prefix)
    if token is None:
        return False
    expected = recorded_at_sort_token(indexed.recorded_at)
    return token == expected and row_key.endswith(f":{indexed.evidence_id}")


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
        raise CausalEvidenceIndexTimestampError(
            "causal evidence index timestamp out of supported range",
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
    "CausalEvidenceIndexTimestampError",
    "DecodedCausalEvidenceIndexV1",
    "DecodedCausalEvidenceIndexV2",
    "decode_causal_evidence_index_v1",
    "decode_causal_evidence_index_v2",
    "encode_causal_evidence_index_v1",
    "encode_causal_evidence_index_v2",
    "execution_index_v1_row_key",
    "execution_index_v1_row_key_prefix",
    "execution_index_v2_row_key",
    "execution_index_v2_row_key_from_evidence",
    "execution_index_v2_row_key_prefix",
    "is_v2_index_row_key",
    "recorded_at_sort_token",
    "transport_index_v1_row_key",
    "transport_index_v1_row_key_prefix",
    "transport_index_v2_row_key",
    "transport_index_v2_row_key_from_evidence",
    "transport_index_v2_row_key_prefix",
    "v2_index_matches_row_key",
]
