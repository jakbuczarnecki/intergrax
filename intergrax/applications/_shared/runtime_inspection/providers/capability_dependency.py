# © Artur Czarnecki. All rights reserved.

"""Capability dependency inspection provider (P1.4)."""

from __future__ import annotations

from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
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
)
from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
    InspectionProvenanceRef,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionExtensionEvidence,
    InspectionProviderContribution,
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.runtime_inspection.scope import InspectionScope
from intergrax.contracts.execution_identity import ExecutionId


_PROVIDER_ID = "capability_dependency"


def _capability_explanations(
    capability_key: str,
    validation: CapabilityDependencyValidationResult,
) -> tuple[InspectionExplanation, ...]:
    explanations: list[InspectionExplanation] = []
    for failure in sorted(
        validation.required_failures,
        key=lambda item: (
            item.owner.canonical_key,
            item.dependency.canonical_key,
            item.requirement.value,
        ),
    ):
        if failure.owner.canonical_key != capability_key:
            continue
        explanations.append(
            InspectionExplanation(
                subject=capability_key,
                facts=(
                    f"dependency={failure.dependency.canonical_key}",
                    f"requirement={failure.requirement.value}",
                    f"status={failure.status.value}",
                    f"source_domains={','.join(sorted(failure.source_domains))}",
                ),
                reasons=(failure.reason,),
                provenance_refs=(
                    InspectionProvenanceRef(
                        kind="dependency_provider",
                        ref=failure.source_domain,
                    ),
                ),
            ),
        )
    for degradation in sorted(
        validation.optional_degradations,
        key=lambda item: (
            item.owner.canonical_key,
            item.dependency.canonical_key,
            item.requirement.value,
        ),
    ):
        if degradation.owner.canonical_key != capability_key:
            continue
        explanations.append(
            InspectionExplanation(
                subject=capability_key,
                facts=(
                    f"dependency={degradation.dependency.canonical_key}",
                    f"requirement={degradation.requirement.value}",
                    f"status={degradation.status.value}",
                    f"source_domains={','.join(sorted(degradation.source_domains))}",
                ),
                reasons=(degradation.reason,),
                provenance_refs=(
                    InspectionProvenanceRef(
                        kind="dependency_provider",
                        ref=degradation.source_domain,
                    ),
                ),
            ),
        )
    return tuple(explanations)


class CapabilityDependencyInspectionProvider:
    """Core provider for P1.3 dependency validation evidence."""

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    def contribute_profile(
        self,
        *,
        resolution: ProfileResolution,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del resolution, configured_profile_ref
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision(
        self,
        *,
        revision_id: EffectiveProfileRevisionId,
        scope: EffectiveProfileRevisionScope,
        revision: EffectiveProfileRevision | None,
    ) -> InspectionProviderContribution:
        del revision_id, scope, revision
        return InspectionProviderContribution(provider_id=self.provider_id)

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
        del tenant_id, execution_id, scope_application_id, scope_tenant_id, binding, pinned_revision
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_capability(
        self,
        *,
        capability_key: str,
        validation: CapabilityDependencyValidationResult,
    ) -> InspectionProviderContribution:
        outcome = next(
            (item for item in validation.outcomes if item.owner.canonical_key == capability_key),
            None,
        )
        payload: dict[str, str] = {}
        if outcome is not None:
            payload["available"] = str(outcome.available)
            payload["degraded"] = str(outcome.degraded)
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=_capability_explanations(capability_key, validation),
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.CAPABILITY,
                    subject=capability_key,
                    payload=payload,
                ),
            )
            if payload
            else (),
        )

    def contribute_revision_compare(
        self,
        *,
        from_revision: EffectiveProfileRevision,
        to_revision: EffectiveProfileRevision,
    ) -> InspectionProviderContribution:
        del from_revision, to_revision
        return InspectionProviderContribution(provider_id=self.provider_id)


def capability_dependency_inspection_provider() -> RuntimeInspectionProvider:
    return CapabilityDependencyInspectionProvider()
