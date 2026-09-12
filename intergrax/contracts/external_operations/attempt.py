# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External operation attempt lifecycle contract (R1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Final
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_task_id,
)
from intergrax.contracts.external_operations.intent import ExternalOperationIntent

SCHEMA_EXTERNAL_OPERATION_ATTEMPT_V1: Final = "external_operation_attempt.v1"


class ExternalOperationAttemptLifecycle(StrEnum):
    CREATED = "CREATED"
    ADMITTED = "ADMITTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TERMINAL = frozenset(
    {
        ExternalOperationAttemptLifecycle.SUCCEEDED,
        ExternalOperationAttemptLifecycle.FAILED,
        ExternalOperationAttemptLifecycle.CANCELLED,
    }
)

_ALLOWED: dict[
    ExternalOperationAttemptLifecycle, frozenset[ExternalOperationAttemptLifecycle]
] = {
    ExternalOperationAttemptLifecycle.CREATED: frozenset(
        {
            ExternalOperationAttemptLifecycle.ADMITTED,
            ExternalOperationAttemptLifecycle.CANCELLED,
        }
    ),
    ExternalOperationAttemptLifecycle.ADMITTED: frozenset(
        {
            ExternalOperationAttemptLifecycle.EXECUTING,
            ExternalOperationAttemptLifecycle.CANCELLED,
        }
    ),
    ExternalOperationAttemptLifecycle.EXECUTING: frozenset(_TERMINAL),
}


class ExternalOperationAttemptTransitionError(RuntimeError):
    """Illegal lifecycle transition (e.g. CREATED → SUCCEEDED)."""


class ExternalOperationAttempt(BaseModel):
    """Durable attempt record — admission required before execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operation_attempt_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("operation_attempt_id", "attempt_id"),
    )
    intent: ExternalOperationIntent
    tenant_id: str | None = Field(default=None, min_length=1)
    task_id: TaskId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    lifecycle: ExternalOperationAttemptLifecycle = (
        ExternalOperationAttemptLifecycle.CREATED
    )
    provider_id: str | None = Field(default=None, min_length=1)
    admitted_at: datetime | None = None
    execution_started_at: datetime | None = None
    terminal_at: datetime | None = None

    @model_validator(mode="after")
    def _forbid_created_to_terminal(self) -> ExternalOperationAttempt:
        if (
            self.lifecycle in _TERMINAL
            and self.admitted_at is None
            and self.lifecycle is not ExternalOperationAttemptLifecycle.CANCELLED
        ):
            raise ValueError(
                "terminal success/failure requires admission (CREATED → terminal forbidden)"
            )
        return self

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object | None) -> TaskId | None:
        if value is None:
            return None
        return validate_task_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_runtime_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object | None) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    def bind_platform_execution(
        self,
        *,
        execution_id: ExecutionId,
        attempt_id: AttemptId,
    ) -> ExternalOperationAttempt:
        if self.execution_id is not None and self.execution_id != execution_id:
            raise ValueError("execution_id already bound to a different runtime execution")
        if self.attempt_id is not None and self.attempt_id != attempt_id:
            raise ValueError("attempt_id already bound to a different runtime attempt")
        return self.model_copy(
            update={
                "execution_id": execution_id,
                "attempt_id": attempt_id,
            },
        )

    def transition(self, target: ExternalOperationAttemptLifecycle) -> ExternalOperationAttempt:
        allowed = _ALLOWED.get(self.lifecycle, frozenset())
        if target not in allowed and self.lifecycle != target:
            raise ExternalOperationAttemptTransitionError(
                f"cannot transition {self.lifecycle} → {target}"
            )
        now = datetime.now(timezone.utc)
        updates: dict[str, object] = {"lifecycle": target}
        if target is ExternalOperationAttemptLifecycle.ADMITTED:
            updates["admitted_at"] = now
        elif target is ExternalOperationAttemptLifecycle.EXECUTING:
            updates["execution_started_at"] = now
        elif target in _TERMINAL:
            updates["terminal_at"] = now
        return self.model_copy(update=updates)

    def bind_provider(self, provider_id: str) -> ExternalOperationAttempt:
        return self.model_copy(update={"provider_id": provider_id})


def mint_external_operation_attempt_id() -> str:
    return f"ext_op_attempt_{uuid4().hex}"
