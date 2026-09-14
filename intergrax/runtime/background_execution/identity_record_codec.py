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

BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1 = "intergrax.bg_exec_identity.v1"
BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2 = "intergrax.bg_exec_identity.v2"

_IDENTITY_RECORD_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class DecodedBackgroundIdentityRecord:
    kind: Literal["complete_v2", "legacy_v1"]
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None


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
