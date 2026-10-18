# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Durable CREATE follow-on record for accept/cancel (PROD-Q-R1-R7-Q F02 Option B)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
from intergrax.contracts.external_work import (
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
)
from intergrax.contracts.validation import compute_sha256_content_digest, validate_content_digest
from intergrax.runtime.attestation.canonical_json import stable_payload_hash

SCHEMA_GOVERNED_EXTERNAL_WORK_CREATE_FOLLOW_ON_V1 = (
    "governed_external_work_create_follow_on.v1"
)
CREATE_ACTION_EXTERNAL_WORK_CREATE = "external_work.create"


class GovernedExternalWorkCreateFollowOnRecordV1(BaseModel):
    """Immutable host fact: typed adapter view required for follow-on mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_external_work_create_follow_on.v1"] = (
        SCHEMA_GOVERNED_EXTERNAL_WORK_CREATE_FOLLOW_ON_V1
    )
    execution_id: ExecutionId
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    tenant_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    create_action: Literal["external_work.create"] = CREATE_ACTION_EXTERNAL_WORK_CREATE
    create_provider_invocation_id: str = Field(min_length=1)
    snapshot: ExternalWorkSnapshot
    provider: ExternalWorkProviderDescriptor | None = None
    snapshot_content_digest: str
    recorded_at: datetime

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("snapshot_content_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        return validate_content_digest(value)

    @field_validator("recorded_at")
    @classmethod
    def _aware_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _identity_and_digest_binding(self) -> GovernedExternalWorkCreateFollowOnRecordV1:
        if str(self.task_id) != str(self.snapshot.correlation.task_id):
            raise ValueError("snapshot task_id mismatch")
        if str(self.run_id) != str(self.snapshot.correlation.run_id):
            raise ValueError("snapshot run_id mismatch")
        expected = compute_snapshot_content_digest(self.snapshot)
        if self.snapshot_content_digest != expected:
            raise ValueError("snapshot_content_digest mismatch")
        return self


def compute_snapshot_content_digest(snapshot: ExternalWorkSnapshot) -> str:
    """Stable digest for follow-on record binding (canonical JSON)."""
    payload = snapshot.model_dump(mode="json")
    return compute_sha256_content_digest(
        stable_payload_hash(payload).encode("utf-8"),
    )


def follow_on_records_equivalent(
    left: GovernedExternalWorkCreateFollowOnRecordV1,
    right: GovernedExternalWorkCreateFollowOnRecordV1,
) -> bool:
    """True when two records carry the same authoritative follow-on facts."""
    return (
        left.execution_id == right.execution_id
        and left.create_provider_invocation_id == right.create_provider_invocation_id
        and left.snapshot_content_digest == right.snapshot_content_digest
    )


def parse_governed_external_work_create_follow_on_record_json(
    raw: str,
) -> GovernedExternalWorkCreateFollowOnRecordV1:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_create_follow_on_record_json") from exc
    if not isinstance(data, dict):
        raise ValueError("create_follow_on_record_must_be_object")
    schema = data.get("schema_version")
    if schema != SCHEMA_GOVERNED_EXTERNAL_WORK_CREATE_FOLLOW_ON_V1:
        raise ValueError(f"unsupported_create_follow_on_schema:{schema!r}")
    try:
        return GovernedExternalWorkCreateFollowOnRecordV1.model_validate(data)
    except ValidationError as exc:
        raise ValueError("invalid_create_follow_on_record_payload") from exc


__all__ = [
    "CREATE_ACTION_EXTERNAL_WORK_CREATE",
    "GovernedExternalWorkCreateFollowOnRecordV1",
    "SCHEMA_GOVERNED_EXTERNAL_WORK_CREATE_FOLLOW_ON_V1",
    "compute_snapshot_content_digest",
    "follow_on_records_equivalent",
    "parse_governed_external_work_create_follow_on_record_json",
]
