# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned durable encoding for background execution identity (NPSC-5F)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.npsc5f_compatibility import Npsc5fCompatibilityError

BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1 = "intergrax.bg_exec_identity.v1"
BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2 = "intergrax.bg_exec_identity.v2"

_IDENTITY_RECORD_SEPARATOR = "\n"


class InvalidBackgroundExecutionIdentityV2RecordError(Npsc5fCompatibilityError):
    """Document v2 partition record is not a complete v2 identity shape."""


class InvalidBackgroundExecutionIdentityV1RecordError(Npsc5fCompatibilityError):
    """Document v1 partition record is not a legacy v1 identity shape."""


@dataclass(frozen=True, slots=True)
class DecodedBackgroundIdentityRecord:
    kind: Literal["complete_v2", "legacy_v1"]
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None


def same_identity_triplet(
    left: DecodedBackgroundIdentityRecord,
    right: DecodedBackgroundIdentityRecord,
) -> bool:
    return (
        left.task_id == right.task_id
        and left.run_id == right.run_id
        and left.attempt_id == right.attempt_id
    )


def same_identity_quadruplet(
    left: DecodedBackgroundIdentityRecord,
    right: DecodedBackgroundIdentityRecord,
) -> bool:
    if left.execution_id is None or right.execution_id is None:
        return False
    return (
        same_identity_triplet(left, right) and left.execution_id == right.execution_id
    )


def _document_identity_field_strings(
    data: dict[str, object],
) -> tuple[str, str, str, object | None]:
    task_raw = data.get("task_id")
    run_raw = data.get("run_id")
    attempt_raw = data.get("attempt_id")
    execution_raw = data.get("execution_id")
    if (
        not isinstance(task_raw, str)
        or not isinstance(run_raw, str)
        or not isinstance(attempt_raw, str)
    ):
        raise ValueError("invalid background execution identity record")
    return task_raw, run_raw, attempt_raw, execution_raw


def decode_document_identity_v2_record(
    data: dict[str, object],
) -> DecodedBackgroundIdentityRecord:
    """Decode a record stored under the v2 document partition (complete v2 only)."""
    task_raw, run_raw, attempt_raw, execution_raw = _document_identity_field_strings(
        data,
    )
    if execution_raw is None:
        raise InvalidBackgroundExecutionIdentityV2RecordError(
            "v2 partition background identity record missing execution_id",
        )
    if not isinstance(execution_raw, str):
        raise InvalidBackgroundExecutionIdentityV2RecordError(
            "v2 partition background identity record has invalid execution_id",
        )
    return DecodedBackgroundIdentityRecord(
        kind="complete_v2",
        task_id=validate_task_id(task_raw),
        run_id=validate_run_id(run_raw),
        attempt_id=validate_attempt_id(attempt_raw),
        execution_id=validate_execution_id(execution_raw),
    )


def decode_document_identity_v1_record(
    data: dict[str, object],
) -> DecodedBackgroundIdentityRecord:
    """Decode a record stored under the v1 document partition (legacy v1 only)."""
    task_raw, run_raw, attempt_raw, execution_raw = _document_identity_field_strings(
        data,
    )
    if execution_raw is not None:
        raise InvalidBackgroundExecutionIdentityV1RecordError(
            "v1 partition background identity record must not include execution_id",
        )
    return DecodedBackgroundIdentityRecord(
        kind="legacy_v1",
        task_id=validate_task_id(task_raw),
        run_id=validate_run_id(run_raw),
        attempt_id=validate_attempt_id(attempt_raw),
        execution_id=None,
    )


def encode_background_identity_v2_record(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> bytes:
    return (
        f"{task_id}{_IDENTITY_RECORD_SEPARATOR}"
        f"{run_id}{_IDENTITY_RECORD_SEPARATOR}"
        f"{attempt_id}{_IDENTITY_RECORD_SEPARATOR}"
        f"{execution_id}"
    ).encode("utf-8")


def decode_background_identity_kv_record(raw: bytes) -> DecodedBackgroundIdentityRecord:
    try:
        parts = raw.decode("utf-8").split(_IDENTITY_RECORD_SEPARATOR)
    except UnicodeDecodeError as exc:
        raise ValueError("invalid background execution identity record") from exc
    if len(parts) == 4:
        task_raw, run_raw, attempt_raw, execution_raw = parts
        return DecodedBackgroundIdentityRecord(
            kind="complete_v2",
            task_id=validate_task_id(task_raw),
            run_id=validate_run_id(run_raw),
            attempt_id=validate_attempt_id(attempt_raw),
            execution_id=validate_execution_id(execution_raw),
        )
    if len(parts) == 3:
        task_raw, run_raw, attempt_raw = parts
        return DecodedBackgroundIdentityRecord(
            kind="legacy_v1",
            task_id=validate_task_id(task_raw),
            run_id=validate_run_id(run_raw),
            attempt_id=validate_attempt_id(attempt_raw),
            execution_id=None,
        )
    raise ValueError("invalid background execution identity record")


def persisted_identity_from_document_record_data(
    data: dict[str, object],
) -> DecodedBackgroundIdentityRecord:
    """Partition-agnostic inference; DocumentStore must use versioned decoders."""
    task_raw, run_raw, attempt_raw, execution_raw = _document_identity_field_strings(
        data,
    )
    if execution_raw is None:
        return DecodedBackgroundIdentityRecord(
            kind="legacy_v1",
            task_id=validate_task_id(task_raw),
            run_id=validate_run_id(run_raw),
            attempt_id=validate_attempt_id(attempt_raw),
            execution_id=None,
        )
    if not isinstance(execution_raw, str):
        raise ValueError("invalid background execution identity record")
    return DecodedBackgroundIdentityRecord(
        kind="complete_v2",
        task_id=validate_task_id(task_raw),
        run_id=validate_run_id(run_raw),
        attempt_id=validate_attempt_id(attempt_raw),
        execution_id=validate_execution_id(execution_raw),
    )
