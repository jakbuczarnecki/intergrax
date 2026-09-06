# © Artur Czarnecki. All rights reserved.

"""Effective capability health projection result (P1.5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_health.fact import (
    CapabilityHealthFact,
    CapabilityHealthReason,
)
from intergrax.applications.contracts.capability_health.status import CapabilityHealthStatus
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)


class CapabilityHealthProviderFailure(BaseModel):
    """Typed evidence when a health provider fails during projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class EffectiveCapabilityHealth(BaseModel):
    """Read-only operational health projection — never runtime authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_capability_health.v1"
    capability: CapabilityRef
    status: CapabilityHealthStatus
    reasons: tuple[CapabilityHealthReason, ...] = Field(default_factory=tuple)
    facts: tuple[CapabilityHealthFact, ...] = Field(default_factory=tuple)
    provenance: tuple[str, ...] = Field(default_factory=tuple)
    provider_failures: tuple[CapabilityHealthProviderFailure, ...] = Field(
        default_factory=tuple,
    )
    effective_profile_revision_id: EffectiveProfileRevisionId | None = None
    effective_profile_fingerprint: str | None = None

    @field_validator("effective_profile_revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId | None:
        if value is None:
            return None
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))
