# © Artur Czarnecki. All rights reserved.

"""Execution pinning inspection provider (P1.4)."""

from __future__ import annotations

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
    InspectionProviderContribution,
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.contracts.execution_identity import ExecutionId


_PROVIDER_ID = "execution_binding"


class ExecutionBindingInspectionProvider:
    """Core provider for execution-to-revision pinning facts."""

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
        if binding is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        assert isinstance(binding, EffectiveProfileExecutionBinding)
        facts = (
            f"tenant_id={tenant_id}",
            f"execution_id={execution_id}",
            f"revision_id={binding.revision_id.value}",
            f"fingerprint={binding.fingerprint}",
        )
        if pinned_revision is not None:
            facts = (
                *facts,
                f"scope.application_id={pinned_revision.scope.application_id}",
                f"scope.tenant_id={pinned_revision.scope.tenant_id}",
            )
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=(
                InspectionExplanation(
                    subject="execution.revision_binding",
                    facts=facts,
                    reasons=("execution is pinned to immutable effective profile revision",),
                    provenance_refs=(
                        InspectionProvenanceRef(
                            kind="execution_binding",
                            ref=f"{tenant_id}:{execution_id}",
                        ),
                        InspectionProvenanceRef(
                            kind="revision",
                            ref=binding.revision_id.value,
                        ),
                    ),
                    related_revision_id=binding.revision_id,
                    related_execution_id=execution_id,
                ),
            ),
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


def execution_binding_inspection_provider() -> RuntimeInspectionProvider:
    return ExecutionBindingInspectionProvider()
