# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Frozen platform_causal_evidence.v1 read models (legacy incomplete correlation)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
)

PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1 = "platform_causal_evidence.v1"


class LegacyRuntimeExecutionRef(BaseModel):
    """Frozen v1 runtime execution reference without ExecutionId."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    tenant_id: str

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id_field(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id_field(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id_field(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("tenant_id")
    @classmethod
    def _require_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        if not value or not value.strip():
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != value.strip():
            raise ValueError(
                "tenant_id must not contain leading or trailing whitespace"
            )
        return value


class LegacyPlatformCausalEvidence(BaseModel):
    """Legacy incomplete causal evidence (no target ExecutionId)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_causal_evidence.v1"] = (
        PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1
    )
    evidence_id: EventId
    relation_kind: CausalRelationKind
    tenant_id: str
    source: MessageBusTaskRef
    target: LegacyRuntimeExecutionRef
    recorded_at: datetime

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _validate_evidence_id_field(cls, value: object) -> EventId:
        return validate_event_id(value)

    @field_validator("tenant_id")
    @classmethod
    def _require_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        if not value or not value.strip():
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != value.strip():
            raise ValueError(
                "tenant_id must not contain leading or trailing whitespace"
            )
        return value

    @model_validator(mode="after")
    def _enforce_tenant_boundary(self) -> LegacyPlatformCausalEvidence:
        if self.source.tenant_id != self.tenant_id:
            raise ValueError("source.tenant_id must match evidence tenant_id")
        if self.target.tenant_id != self.tenant_id:
            raise ValueError("target.tenant_id must match evidence tenant_id")
        return self
