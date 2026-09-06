# © Artur Czarnecki. All rights reserved.

"""Domain-owned capability health provider seam (P1.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.applications.contracts.capability_health.fact import CapabilityHealthFact
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.skills.registry.runtime import SkillRegistry


@dataclass(frozen=True, slots=True)
class CapabilityHealthProjectionContext:
    """Immutable inputs for one capability health projection."""

    capability: CapabilityRef
    validation: CapabilityDependencyValidationResult | None = None
    environment_profile: ApplicationEnvironmentProfile | None = None
    scope_application_id: str | None = None
    scope_tenant_id: str | None = None
    effective_profile_revision_id: EffectiveProfileRevisionId | None = None
    effective_profile_fingerprint: str | None = None
    skill_registry: SkillRegistry | None = None


class CapabilityHealthProvider(Protocol):
    """Domain plugin: emit health facts for one capability projection."""

    @property
    def provider_id(self) -> str:
        """Stable routing identity — distinct from evidence provenance."""

    @property
    def source_provenance(self) -> str:
        """Provenance label for facts emitted by this provider."""

    def health_facts_for(
        self,
        context: CapabilityHealthProjectionContext,
    ) -> tuple[CapabilityHealthFact, ...]:
        """Return health facts owned by this domain for the given capability."""
