# © Artur Czarnecki. All rights reserved.

"""Domain-owned capability dependency provider seam (P1.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.applications.contracts.capability_dependency.dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.skills.registry.runtime import SkillRegistry


@dataclass(frozen=True, slots=True)
class CapabilityDependencyValidationContext:
    """Immutable inputs for declaration and availability evaluation."""

    environment_profile: ApplicationEnvironmentProfile
    skill_registry: SkillRegistry | None = None


class CapabilityDependencyProvider(Protocol):
    """Domain plugin: declare edges and evaluate availability for its declarations."""

    @property
    def provider_id(self) -> str:
        """Stable routing identity for evaluation dispatch."""

    @property
    def source_domain(self) -> str:
        """Provenance label for declarations emitted by this provider."""

    def dependencies_for(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependency, ...]:
        """Return explicit dependency declarations owned by this domain."""

    def evaluate_availability(
        self,
        dependency: CapabilityDependency,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependencyAvailabilityStatus, str]:
        """Evaluate one declaration previously emitted by this provider."""
