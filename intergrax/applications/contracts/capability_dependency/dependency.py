# © Artur Czarnecki. All rights reserved.

"""Typed capability dependency declarations (P1.3)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CapabilityDependencyKind(StrEnum):
    """Domain-justified dependency identity kinds — extend when domains adopt P1.3."""

    SKILL = "skill"
    TOOL = "tool"


class CapabilityDependencyRequirement(StrEnum):
    """Whether a missing dependency blocks or degrades the owner capability."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class CapabilityDependencyAvailabilityStatus(StrEnum):
    """Evaluation outcome for one declared dependency edge."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CapabilityRef(BaseModel):
    """Stable canonical capability identity reused from domain ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: CapabilityDependencyKind
    capability_id: str = Field(min_length=1)

    @property
    def canonical_key(self) -> str:
        return f"{self.kind.value}:{self.capability_id}"


class CapabilityDependency(BaseModel):
    """One explicit owner → dependency edge with provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: CapabilityRef
    dependency: CapabilityRef
    requirement: CapabilityDependencyRequirement
    source_domains: tuple[str, ...] = Field(min_length=1)

    @property
    def dedup_key(self) -> tuple[str, str, str]:
        return (
            self.owner.canonical_key,
            self.dependency.kind.value,
            self.dependency.capability_id,
        )
