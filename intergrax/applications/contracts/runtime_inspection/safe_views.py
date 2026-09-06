# © Artur Czarnecki. All rights reserved.

"""Safe inspection serialization projections (P1.4A)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.profile_resolution.decision import (
    DegradedCapability,
    ProfileDependencyFailure,
    ProfileResolutionDecision,
    ProfileResolutionWarning,
)
from intergrax.applications.contracts.profile_resolution.diff import ProfileDiffChangeKind
from intergrax.applications.contracts.profile_resolution.layer import ProfileLayer
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)


class RedactedProfileSnapshot(BaseModel):
    """Redacted effective profile suitable for operator serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[str, Any] = Field(default_factory=dict)


class SafeProfileLayerView(BaseModel):
    """Layer contribution with redacted delta payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: ProfileLayer
    revision: str | None = None
    redacted_delta: dict[str, Any] | None = None


class SafeProfileResolutionView(BaseModel):
    """Serialized profile resolution evidence without raw effective profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str
    effective_profile: RedactedProfileSnapshot
    layers: tuple[SafeProfileLayerView, ...] = Field(default_factory=tuple)
    decisions: tuple[ProfileResolutionDecision, ...] = Field(default_factory=tuple)
    warnings: tuple[ProfileResolutionWarning, ...] = Field(default_factory=tuple)
    dependency_failures: tuple[ProfileDependencyFailure, ...] = Field(default_factory=tuple)
    degraded_capabilities: tuple[DegradedCapability, ...] = Field(default_factory=tuple)


class SafeEffectiveProfileRevisionView(BaseModel):
    """Serialized revision snapshot without raw canonical objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision_id: EffectiveProfileRevisionId
    fingerprint: str = Field(min_length=1)
    scope: EffectiveProfileRevisionScope
    predecessor_revision_id: EffectiveProfileRevisionId | None = None
    effective_profile: RedactedProfileSnapshot
    resolution: SafeProfileResolutionView


class SafeProfileDiffEntryView(BaseModel):
    """One semantic diff entry with redacted before/after values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    before: str | None
    after: str | None
    change_kind: ProfileDiffChangeKind


class SafeEffectiveProfileDiffView(BaseModel):
    """Semantic diff safe for direct operator serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "effective_profile_diff.v1"
    from_revision_id: EffectiveProfileRevisionId
    to_revision_id: EffectiveProfileRevisionId
    from_fingerprint: str
    to_fingerprint: str
    entries: tuple[SafeProfileDiffEntryView, ...] = Field(default_factory=tuple)
