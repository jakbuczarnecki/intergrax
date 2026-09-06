# © Artur Czarnecki. All rights reserved.

"""Inspection subject references (P1.4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


class ProfileInspectionRef(BaseModel):
    """Profile inspection subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configured_profile_ref: str | None = None


class RevisionInspectionRef(BaseModel):
    """Revision inspection subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision_id: EffectiveProfileRevisionId
    scope: EffectiveProfileRevisionScope

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class ExecutionInspectionRef(BaseModel):
    """Execution inspection subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId
    scope_application_id: str = Field(min_length=1)
    scope_tenant_id: str | None = None

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)


class CapabilityInspectionRef(BaseModel):
    """Capability dependency inspection subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: CapabilityRef
