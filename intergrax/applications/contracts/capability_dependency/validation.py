# © Artur Czarnecki. All rights reserved.

"""Capability dependency validation evidence (P1.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.capability_dependency.dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyRequirement,
    CapabilityRef,
)


class CapabilityDependencyEvaluation(BaseModel):
    """One evaluated dependency edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dependency: CapabilityDependency
    status: CapabilityDependencyAvailabilityStatus
    reason: str


class CapabilityDependencyFailureEvidence(BaseModel):
    """Typed evidence for a required dependency that blocks execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: CapabilityRef
    dependency: CapabilityRef
    dependency_kind: CapabilityDependencyKind
    requirement: CapabilityDependencyRequirement
    status: CapabilityDependencyAvailabilityStatus
    reason: str
    source_domains: tuple[str, ...] = Field(min_length=1)
    source_domain: str = Field(min_length=1)


class CapabilityDependencyDegradationEvidence(BaseModel):
    """Typed evidence for optional dependency loss — no silent fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: CapabilityRef
    dependency: CapabilityRef
    dependency_kind: CapabilityDependencyKind
    requirement: CapabilityDependencyRequirement
    status: CapabilityDependencyAvailabilityStatus
    reason: str
    source_domains: tuple[str, ...] = Field(min_length=1)
    source_domain: str = Field(min_length=1)


class CapabilityDependencyOutcome(BaseModel):
    """Per-owner capability rollup from evaluated dependency edges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: CapabilityRef
    available: bool
    degraded: bool
    evaluations: tuple[CapabilityDependencyEvaluation, ...] = Field(default_factory=tuple)


class CapabilityDependencyValidationResult(BaseModel):
    """Deterministic aggregate from declarations and availability facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    declarations: tuple[CapabilityDependency, ...] = Field(default_factory=tuple)
    evaluations: tuple[CapabilityDependencyEvaluation, ...] = Field(default_factory=tuple)
    outcomes: tuple[CapabilityDependencyOutcome, ...] = Field(default_factory=tuple)
    required_failures: tuple[CapabilityDependencyFailureEvidence, ...] = Field(
        default_factory=tuple,
    )
    optional_degradations: tuple[CapabilityDependencyDegradationEvidence, ...] = Field(
        default_factory=tuple,
    )

    @property
    def available(self) -> bool:
        return not self.required_failures

    @property
    def degraded(self) -> bool:
        return bool(self.optional_degradations)
