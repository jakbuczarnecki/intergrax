# © Artur Czarnecki. All rights reserved.

"""Execution pinning evidence for effective profile revisions (P1.2)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)


EFFECTIVE_PROFILE_REVISION_METADATA_KEY = "effective_profile_revision.v1"


class EffectiveProfileRevisionCheckpointEvidence(BaseModel):
    """Checkpoint/resume evidence reference — not revision authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_revision_checkpoint.v1"
    revision_id: EffectiveProfileRevisionId
    fingerprint: str = Field(min_length=1)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class EffectiveProfileExecutionBinding(BaseModel):
    """Immutable execution-to-revision admission evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_execution_binding.v1"
    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId
    revision_id: EffectiveProfileRevisionId
    fingerprint: str = Field(min_length=1)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class EffectiveProfileExecutionPinningStore(Protocol):
    """Durable execution pinning for admitted work."""

    def pin(self, binding: EffectiveProfileExecutionBinding) -> None:
        """Record one immutable execution revision binding."""

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> EffectiveProfileExecutionBinding | None:
        """Resolve pinned revision for one execution."""
