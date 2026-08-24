# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical cross-boundary causal evidence (DIAG-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    mint_event_id,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)

PLATFORM_CAUSAL_EVIDENCE_SCHEMA = "platform_causal_evidence.v1"


class CausalRelationKind(StrEnum):
    """Minimal causal semantics for DIAG-1."""

    TRANSPORT_TASK_TRIGGERED_EXECUTION = "transport_task.triggered_execution"


class MessageBusTaskRef(BaseModel):
    """Provider-neutral async transport task identity — not runtime ``TaskId``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    task_id: str
    tenant_id: str

    @field_validator("provider", "task_id", "tenant_id")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError(f"{cls.__name__} fields must be str")
        if not value or not value.strip():
            raise ValueError("field must be non-empty and not whitespace-only")
        if value != value.strip():
            raise ValueError("field must not contain leading or trailing whitespace")
        return value

    @field_validator("task_id")
    @classmethod
    def _reject_canonical_runtime_task_id(cls, value: str) -> str:
        try:
            validate_task_id(value)
        except (TypeError, ValueError):
            return value
        raise ValueError(
            "MessageBus task_id must not use canonical runtime TaskId format",
        )


class RuntimeExecutionRef(BaseModel):
    """Canonical runtime execution identity at the transport→execution boundary."""

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
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return value


class PlatformCausalEvidence(BaseModel):
    """Immutable causal fact linking async transport to runtime execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_causal_evidence.v1"] = PLATFORM_CAUSAL_EVIDENCE_SCHEMA
    evidence_id: EventId = Field(default_factory=mint_event_id)
    relation_kind: CausalRelationKind
    tenant_id: str
    source: MessageBusTaskRef
    target: RuntimeExecutionRef
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def _enforce_tenant_boundary(self) -> PlatformCausalEvidence:
        if self.source.tenant_id != self.tenant_id:
            raise ValueError("source.tenant_id must match evidence tenant_id")
        if self.target.tenant_id != self.tenant_id:
            raise ValueError("target.tenant_id must match evidence tenant_id")
        return self
