# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical runtime inspection query (INSPECT-01-A)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id

DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT = 256
MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT = 4096


class RuntimeInspectionQuery(BaseModel):
    """Primary operator lookup: tenant + canonical ExecutionId."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "runtime_inspection_query.v1"
    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId
    timeline_limit: int = DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("tenant_id must be non-empty canonical tenant scope")
        return value

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("timeline_limit")
    @classmethod
    def _validate_timeline_limit(cls, value: int) -> int:
        if type(value) is not int or isinstance(value, bool):
            raise TypeError("timeline_limit must be int")
        if value < 1 or value > MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT:
            raise ValueError("timeline_limit out of bounds")
        return value


__all__ = [
    "DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "RuntimeInspectionQuery",
]
