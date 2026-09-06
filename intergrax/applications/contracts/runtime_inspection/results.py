# © Artur Czarnecki. All rights reserved.

"""Runtime inspection read-model results (P1.4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyDegradationEvidence,
    CapabilityDependencyFailureEvidence,
    CapabilityDependencyOutcome,
    CapabilityDependencyValidationResult,
)
from intergrax.applications.contracts.profile_resolution.diff import EffectiveProfileDiff
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    validate_effective_profile_revision_id,
)
from intergrax.applications.contracts.runtime_inspection.completeness import (
    InspectionCompleteness,
)
from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
)
from intergrax.applications.contracts.runtime_inspection.inconsistency import (
    InspectionInconsistency,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionExtensionEvidence,
    InspectionProviderFailure,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


class ProfileInspectionResult(BaseModel):
    """Read-only profile resolution explainability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "profile_inspection.v1"
    configured_profile_ref: str | None = None
    resolution: ProfileResolution
    completeness: InspectionCompleteness
    inconsistencies: tuple[InspectionInconsistency, ...] = Field(default_factory=tuple)
    explanations: tuple[InspectionExplanation, ...] = Field(default_factory=tuple)
    provider_failures: tuple[InspectionProviderFailure, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)


class RevisionInspectionResult(BaseModel):
    """Read-only effective profile revision snapshot inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "revision_inspection.v1"
    revision_id: EffectiveProfileRevisionId
    scope: EffectiveProfileRevisionScope
    revision: EffectiveProfileRevision | None = None
    completeness: InspectionCompleteness
    inconsistencies: tuple[InspectionInconsistency, ...] = Field(default_factory=tuple)
    explanations: tuple[InspectionExplanation, ...] = Field(default_factory=tuple)
    provider_failures: tuple[InspectionProviderFailure, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> EffectiveProfileRevisionId:
        if isinstance(value, EffectiveProfileRevisionId):
            return value
        return EffectiveProfileRevisionId(validate_effective_profile_revision_id(value))


class RevisionCompareResult(BaseModel):
    """Semantic diff between two revisions — reuses P1.2 diff engine output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "revision_compare.v1"
    diff: EffectiveProfileDiff
    completeness: InspectionCompleteness
    inconsistencies: tuple[InspectionInconsistency, ...] = Field(default_factory=tuple)
    provider_failures: tuple[InspectionProviderFailure, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)


class ExecutionInspectionResult(BaseModel):
    """Execution pinning inspection — exact pinned revision, never current host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "execution_inspection.v1"
    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId
    scope_application_id: str = Field(min_length=1)
    scope_tenant_id: str | None = None
    binding: EffectiveProfileExecutionBinding | None = None
    pinned_revision: EffectiveProfileRevision | None = None
    completeness: InspectionCompleteness
    inconsistencies: tuple[InspectionInconsistency, ...] = Field(default_factory=tuple)
    explanations: tuple[InspectionExplanation, ...] = Field(default_factory=tuple)
    provider_failures: tuple[InspectionProviderFailure, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)


class CapabilityInspectionResult(BaseModel):
    """Capability dependency validation explainability — not operational readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "capability_inspection.v1"
    capability: CapabilityRef
    validation: CapabilityDependencyValidationResult
    outcome: CapabilityDependencyOutcome | None = None
    required_failures: tuple[CapabilityDependencyFailureEvidence, ...] = Field(
        default_factory=tuple,
    )
    optional_degradations: tuple[CapabilityDependencyDegradationEvidence, ...] = Field(
        default_factory=tuple,
    )
    completeness: InspectionCompleteness
    inconsistencies: tuple[InspectionInconsistency, ...] = Field(default_factory=tuple)
    explanations: tuple[InspectionExplanation, ...] = Field(default_factory=tuple)
    provider_failures: tuple[InspectionProviderFailure, ...] = Field(default_factory=tuple)
    extension_evidence: tuple[InspectionExtensionEvidence, ...] = Field(default_factory=tuple)
