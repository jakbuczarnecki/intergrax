# © Artur Czarnecki. All rights reserved.

"""Profile resolution evidence records (P1.1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.profile_resolution.delta import ProfileDelta
from intergrax.applications.contracts.profile_resolution.layer import ProfileLayer


class ProfileResolutionDecisionKind(StrEnum):
    """Typed resolution outcome for one field path."""

    APPLIED = "applied"
    REJECTED = "rejected"
    CLAMPED = "clamped"
    UNCHANGED = "unchanged"
    DEGRADED = "degraded"


class ProfileResolutionDecision(BaseModel):
    """Evidence for one meaningful resolution step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    requested_value: str | None
    source_layer: ProfileLayer
    previous_value: str | None
    decision: ProfileResolutionDecisionKind
    effective_value: str | None
    reason: str


class ProfileResolutionWarning(BaseModel):
    """Non-fatal resolution evidence — does not change effective configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    message: str
    source_layer: ProfileLayer | None = None


class ProfileDependencyFailure(BaseModel):
    """Placeholder carrier for P1.3 dependency validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str
    reason: str
    source_layer: ProfileLayer | None = None


class DegradedCapability(BaseModel):
    """Placeholder carrier for degraded capability state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str
    reason: str
    source_layer: ProfileLayer | None = None


class ProfileLayerResolution(BaseModel):
    """Immutable record of one layer contribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: ProfileLayer
    revision: str | None = None
    delta: ProfileDelta | None = None
