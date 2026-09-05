# © Artur Czarnecki. All rights reserved.

"""Effective profile revision snapshot contract (P1.2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)


class EffectiveProfileRevisionScope(BaseModel):
    """Tenant-aware revision ownership for historical lookup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = Field(min_length=1)
    tenant_id: str | None = None


class EffectiveProfileRevision(BaseModel):
    """Immutable admitted effective profile snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_revision.v1"
    revision_id: EffectiveProfileRevisionId
    fingerprint: str = Field(min_length=1)
    effective_profile: ApplicationEnvironmentProfile
    resolution: ProfileResolution
    scope: EffectiveProfileRevisionScope
    predecessor_revision_id: EffectiveProfileRevisionId | None = None

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))

    @field_validator("predecessor_revision_id", mode="before")
    @classmethod
    def _validate_predecessor_revision_id(
        cls,
        value: object,
    ) -> EffectiveProfileRevisionId | None:
        if value is None:
            return None
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))
