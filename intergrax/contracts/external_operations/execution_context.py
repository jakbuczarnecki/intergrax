# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform execution binding for external operations (DIAG integration R2)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_task_id,
)

SCHEMA_EXTERNAL_OPERATION_EXECUTION_CONTEXT_V1: Final = (
    "external_operation_execution_context.v1"
)


class ExternalOperationExecutionContext(BaseModel):
    """Full platform identity for one admitted external operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: ExecutionId
    attempt_id: AttemptId
    tenant_id: str = Field(min_length=1)
    task_id: TaskId
    operation_intent_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    scope: str = Field(min_length=1, max_length=512)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)


__all__ = [
    "ExternalOperationExecutionContext",
    "SCHEMA_EXTERNAL_OPERATION_EXECUTION_CONTEXT_V1",
]
