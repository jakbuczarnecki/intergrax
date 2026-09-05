# © Artur Czarnecki. All rights reserved.

"""ProfileResolution read-model contract (P1.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution.decision import (
    DegradedCapability,
    ProfileDependencyFailure,
    ProfileLayerResolution,
    ProfileResolutionDecision,
    ProfileResolutionWarning,
)


class ProfileResolution(BaseModel):
    """Immutable derived evidence for configured → effective profile resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_profile: ApplicationEnvironmentProfile
    layers: tuple[ProfileLayerResolution, ...] = Field(default_factory=tuple)
    decisions: tuple[ProfileResolutionDecision, ...] = Field(default_factory=tuple)
    warnings: tuple[ProfileResolutionWarning, ...] = Field(default_factory=tuple)
    dependency_failures: tuple[ProfileDependencyFailure, ...] = Field(default_factory=tuple)
    degraded_capabilities: tuple[DegradedCapability, ...] = Field(default_factory=tuple)
    fingerprint: str
