# © Artur Czarnecki. All rights reserved.

"""Deterministic encoding for durable execution deadline authority."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from intergrax.contracts.execution_deadline.snapshot import (
    EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
    ExecutionDeadlineAuthoritySnapshot,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.execution.deadline_authority.persistence import (
    ExecutionDeadlineCodecError,
)


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ExecutionDeadlineCodecError("datetime must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat()


def _decode_datetime(raw: object, field: str) -> datetime:
    if not isinstance(raw, str) or not raw:
        raise ExecutionDeadlineCodecError(f"invalid {field}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ExecutionDeadlineCodecError(f"invalid {field}") from exc
    if parsed.tzinfo is None:
        raise ExecutionDeadlineCodecError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def encode_execution_deadline_authority_snapshot(
    snapshot: ExecutionDeadlineAuthoritySnapshot,
) -> bytes:
    payload = {
        "schema_version": snapshot.schema_version,
        "run_id": str(snapshot.run_id),
        "deadline_at_utc": (
            _encode_datetime(snapshot.deadline_at_utc)
            if snapshot.deadline_at_utc is not None
            else None
        ),
        "authority_created_at_utc": _encode_datetime(snapshot.authority_created_at_utc),
        "policy_max_wall_time_seconds": snapshot.policy_max_wall_time_seconds,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_execution_deadline_authority_snapshot(raw: bytes) -> ExecutionDeadlineAuthoritySnapshot:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionDeadlineCodecError("invalid execution deadline record encoding") from exc
    if not isinstance(payload, dict):
        raise ExecutionDeadlineCodecError("invalid execution deadline record payload")
    schema_version = payload.get("schema_version")
    if schema_version != EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION:
        raise ExecutionDeadlineCodecError("unsupported execution deadline schema version")
    run_id = validate_run_id(payload.get("run_id"))
    deadline_raw = payload.get("deadline_at_utc")
    deadline_at_utc = (
        _decode_datetime(deadline_raw, "deadline_at_utc") if deadline_raw is not None else None
    )
    authority_created_at_utc = _decode_datetime(
        payload.get("authority_created_at_utc"),
        "authority_created_at_utc",
    )
    policy_raw = payload.get("policy_max_wall_time_seconds")
    policy_max_wall_time_seconds: float | None
    if policy_raw is None:
        policy_max_wall_time_seconds = None
    elif isinstance(policy_raw, (int, float)) and not isinstance(policy_raw, bool):
        policy_max_wall_time_seconds = float(policy_raw)
        if policy_max_wall_time_seconds <= 0:
            raise ExecutionDeadlineCodecError("policy_max_wall_time_seconds must be positive")
    else:
        raise ExecutionDeadlineCodecError("invalid policy_max_wall_time_seconds")
    return ExecutionDeadlineAuthoritySnapshot(
        schema_version=EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
        run_id=RunId(run_id),
        deadline_at_utc=deadline_at_utc,
        authority_created_at_utc=authority_created_at_utc,
        policy_max_wall_time_seconds=policy_max_wall_time_seconds,
    )
