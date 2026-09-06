# © Artur Czarnecki. All rights reserved.

"""Runtime inspection provider extension seam (P1.4)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
)
from intergrax.applications.contracts.runtime_inspection.scope import InspectionScope
from intergrax.contracts.execution_identity import ExecutionId


class InspectionProviderFailure(BaseModel):
    """Typed optional provider failure — core facts remain visible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class InspectionExtensionEvidence(BaseModel):
    """Typed evidence appended by extension providers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    scope: InspectionScope
    subject: str = Field(min_length=1)
    payload: dict[str, str] = Field(default_factory=dict)


class InspectionProviderContribution(BaseModel):
    """Merged contribution from one inspection provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    explanations: tuple[InspectionExplanation, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)
    failure: InspectionProviderFailure | None = None


class RuntimeInspectionProvider(Protocol):
    """Domain plugin: append read-only inspection facts without mutating runtime."""

    @property
    def provider_id(self) -> str:
        """Stable provider identity — distinct from provenance labels."""

    def contribute_profile(
        self,
        *,
        resolution: ProfileResolution,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        """Optional profile-scope evidence."""

    def contribute_revision(
        self,
        *,
        revision_id: EffectiveProfileRevisionId,
        scope: EffectiveProfileRevisionScope,
        revision: EffectiveProfileRevision | None,
    ) -> InspectionProviderContribution:
        """Optional revision-scope evidence."""

    def contribute_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        scope_application_id: str,
        scope_tenant_id: str | None,
        binding: object | None,
        pinned_revision: EffectiveProfileRevision | None,
    ) -> InspectionProviderContribution:
        """Optional execution-scope evidence."""

    def contribute_capability(
        self,
        *,
        capability_key: str,
        validation: CapabilityDependencyValidationResult,
    ) -> InspectionProviderContribution:
        """Optional capability-scope evidence."""

    def contribute_revision_compare(
        self,
        *,
        from_revision: EffectiveProfileRevision,
        to_revision: EffectiveProfileRevision,
    ) -> InspectionProviderContribution:
        """Optional revision-compare evidence."""
