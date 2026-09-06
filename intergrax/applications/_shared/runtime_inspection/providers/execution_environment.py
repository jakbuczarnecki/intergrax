# © Artur Czarnecki. All rights reserved.

"""Execution environment inspection provider (P1.8)."""

from __future__ import annotations

import json

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
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.sandbox.enforcement import resolve_inspection_execution_environment
from intergrax.runtime.sandbox.execution_environment import (
    EffectiveExecutionEnvironment,
    ExecutionEnvironmentResolutionFailureReason,
)
from intergrax.runtime.sandbox.resolver import profile_isolation_authority


_PROVIDER_ID = "execution_environment"


def _payload_from_environment(
    environment: EffectiveExecutionEnvironment | None,
    *,
    failure_reason: ExecutionEnvironmentResolutionFailureReason | None = None,
    failure_message: str | None = None,
) -> dict[str, str]:
    if environment is not None:
        return {
            "status": "resolved",
            "provider_id": environment.provider_ref.provider_id,
            "provider_kind": environment.provider_ref.provider_kind.value,
            "sandbox_required": str(environment.sandbox_required),
            "filesystem_access": environment.filesystem_access.value,
            "network_access": environment.network_access.value,
            "process_execution": environment.process_execution.value,
            "provenance": json.dumps(environment.provenance.model_dump(mode="json"), sort_keys=True),
        }
    return {
        "status": "unavailable",
        "failure_reason": failure_reason.value if failure_reason is not None else "unknown",
        "failure_message": failure_message or "execution environment unavailable",
    }


class ExecutionEnvironmentInspectionProvider:
    """Projects safe effective execution environment from pinned revision."""

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
        authority = profile_isolation_authority(resolution.effective_profile)
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.PROFILE,
                    subject="execution_environment_authority",
                    payload={
                        "sandbox_configured": str(authority.sandbox_configured),
                        "filesystem_access": authority.filesystem_access.value,
                        "network_access": authority.network_access.value,
                        "process_execution": authority.process_execution.value,
                    },
                ),
            ),
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
        resolution = resolve_inspection_execution_environment(revision=revision)
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.PROFILE,
                    subject="revision_execution_environment",
                    payload=_payload_from_environment(
                        resolution.environment,
                        failure_reason=resolution.failure.reason if resolution.failure else None,
                        failure_message=resolution.failure.message if resolution.failure else None,
                    ),
                ),
            ),
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
            return InspectionProviderContribution(
                provider_id=self.provider_id,
                explanations=(
                    InspectionExplanation(
                        subject="execution_environment",
                        facts=("pinned_revision=missing",),
                        reasons=("execution environment requires pinned revision",),
                        provenance_refs=(
                            InspectionProvenanceRef(
                                kind="execution_environment",
                                ref="unavailable",
                            ),
                        ),
                    ),
                ),
            )
        resolution = resolve_inspection_execution_environment(revision=pinned_revision)
        payload = _payload_from_environment(
            resolution.environment,
            failure_reason=resolution.failure.reason if resolution.failure else None,
            failure_message=resolution.failure.message if resolution.failure else None,
        )
        facts = (
            f"revision_id={pinned_revision.revision_id.value}",
            f"status={payload['status']}",
        )
        if resolution.environment is not None:
            facts = (
                *facts,
                f"provider_id={resolution.environment.provider_ref.provider_id}",
                f"sandbox_required={resolution.environment.sandbox_required}",
            )
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.EXECUTION,
                    subject="execution_effective_environment",
                    payload=payload,
                ),
            ),
            explanations=(
                InspectionExplanation(
                    subject="execution_environment",
                    facts=facts,
                    reasons=(
                        resolution.failure.message
                        if resolution.failure is not None
                        else "effective execution environment resolved from pinned revision",
                    ),
                    provenance_refs=(
                        InspectionProvenanceRef(
                            kind="effective_profile_revision",
                            ref=pinned_revision.revision_id.value,
                        ),
                    ),
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


def execution_environment_inspection_provider() -> RuntimeInspectionProvider:
    return ExecutionEnvironmentInspectionProvider()
