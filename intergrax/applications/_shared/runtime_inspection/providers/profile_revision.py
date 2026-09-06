# © Artur Czarnecki. All rights reserved.

"""Profile and revision inspection provider (P1.4)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution.decision import ProfileResolutionDecision
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
    InspectionProvenanceRef,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionProviderContribution,
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.runtime_inspection.scope import InspectionScope
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.contracts.execution_identity import ExecutionId


_PROVIDER_ID = "profile_revision"


def _sorted_decisions(
    decisions: tuple[ProfileResolutionDecision, ...],
) -> tuple[ProfileResolutionDecision, ...]:
    return tuple(sorted(decisions, key=lambda item: (item.path, item.source_layer.value)))


def _profile_explanations(resolution: ProfileResolution) -> tuple[InspectionExplanation, ...]:
    explanations: list[InspectionExplanation] = []
    for index, decision in enumerate(_sorted_decisions(resolution.decisions)):
        explanations.append(
            InspectionExplanation(
                subject=decision.path,
                facts=(
                    f"requested={decision.requested_value}",
                    f"previous={decision.previous_value}",
                    f"effective={decision.effective_value}",
                ),
                reasons=(decision.reason,),
                provenance_refs=(
                    InspectionProvenanceRef(
                        kind="profile_resolution_decision",
                        ref=str(index),
                    ),
                    InspectionProvenanceRef(
                        kind="profile_layer",
                        ref=decision.source_layer.value,
                    ),
                ),
            ),
        )
    return tuple(explanations)


class ProfileRevisionInspectionProvider:
    """Core provider for profile resolution and revision facts."""

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    def contribute_profile(
        self,
        *,
        resolution: ProfileResolution,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del configured_profile_ref
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=_profile_explanations(resolution),
        )

    def contribute_revision(
        self,
        *,
        revision_id: EffectiveProfileRevisionId,
        scope: EffectiveProfileRevisionScope,
        revision: EffectiveProfileRevision | None,
    ) -> InspectionProviderContribution:
        del revision_id, scope
        if revision is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=_profile_explanations(revision.resolution),
        )

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
        del tenant_id, execution_id, scope_application_id, scope_tenant_id, binding
        if pinned_revision is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=_profile_explanations(pinned_revision.resolution),
        )

    def contribute_capability(
        self,
        *,
        capability_key: str,
        validation: CapabilityDependencyValidationResult,
    ) -> InspectionProviderContribution:
        del capability_key, validation
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(
        self,
        *,
        from_revision: EffectiveProfileRevision,
        to_revision: EffectiveProfileRevision,
    ) -> InspectionProviderContribution:
        del from_revision, to_revision
        return InspectionProviderContribution(provider_id=self.provider_id)


def profile_revision_inspection_provider() -> RuntimeInspectionProvider:
    return ProfileRevisionInspectionProvider()
