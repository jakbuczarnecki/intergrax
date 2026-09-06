# © Artur Czarnecki. All rights reserved.

"""Typed capability health facts and reasons (P1.5)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef


class CapabilityHealthFactStatus(StrEnum):
    """Evaluation outcome for one health condition fact."""

    SATISFIED = "satisfied"
    DEGRADED = "degraded"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


class CapabilityHealthConditionKind(StrEnum):
    """Semantic condition identity — extend via domain providers."""

    DEPENDENCY_REQUIRED = "dependency.required"
    DEPENDENCY_OPTIONAL = "dependency.optional"
    TOOL_EFFECTIVE_AVAILABILITY = "tool.effective_availability"
    PROVIDER_FAILURE = "provider.failure"


class CapabilityHealthReason(BaseModel):
    """Structured reason — natural-language detail is optional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str = Field(min_length=1)
    source: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    detail: str | None = None


class CapabilityHealthFact(BaseModel):
    """One canonical health fact contributing to effective projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: CapabilityRef
    source: str = Field(min_length=1)
    condition_kind: CapabilityHealthConditionKind
    condition_ref: str = Field(min_length=1)
    scope_application_id: str | None = None
    scope_tenant_id: str | None = None
    status: CapabilityHealthFactStatus
    blocking: bool
    reason: CapabilityHealthReason
    provider_id: str = Field(min_length=1)

    @property
    def semantic_key(self) -> tuple[str, str, str, str | None, str | None]:
        return (
            self.capability.canonical_key,
            self.condition_kind.value,
            self.condition_ref,
            self.scope_application_id,
            self.scope_tenant_id,
        )
