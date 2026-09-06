# © Artur Czarnecki. All rights reserved.

"""Safe capability health views for serialization (P1.5 / P1.4A)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_health.status import CapabilityHealthStatus
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)


class SafeCapabilityHealthReasonView(BaseModel):
    """Sanitized health reason safe for direct serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str = Field(min_length=1)
    source: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    detail: str | None = None


class SafeCapabilityHealthFactView(BaseModel):
    """Sanitized health fact safe for direct serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: CapabilityRef
    source: str = Field(min_length=1)
    condition_kind: str = Field(min_length=1)
    condition_ref: str = Field(min_length=1)
    scope_application_id: str | None = None
    scope_tenant_id: str | None = None
    status: str = Field(min_length=1)
    blocking: bool
    reason: SafeCapabilityHealthReasonView
    provider_id: str = Field(min_length=1)


class SafeCapabilityHealthProviderFailureView(BaseModel):
    """Sanitized provider failure safe for direct serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SafeEffectiveCapabilityHealthView(BaseModel):
    """Serialized effective health projection without raw secret leakage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_capability_health.v1"
    capability: CapabilityRef
    status: CapabilityHealthStatus
    reasons: tuple[SafeCapabilityHealthReasonView, ...] = Field(default_factory=tuple)
    facts: tuple[SafeCapabilityHealthFactView, ...] = Field(default_factory=tuple)
    provenance: tuple[str, ...] = Field(default_factory=tuple)
    provider_failures: tuple[SafeCapabilityHealthProviderFailureView, ...] = Field(
        default_factory=tuple,
    )
    effective_profile_revision_id: EffectiveProfileRevisionId | None = None
    effective_profile_fingerprint: str | None = None
