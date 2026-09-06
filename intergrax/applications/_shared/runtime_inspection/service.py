# © Artur Czarnecki. All rights reserved.

"""Canonical runtime inspection read model (P1.4)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications._shared.capability_health.composition import (
    default_capability_health_providers,
)
from intergrax.applications._shared.capability_health.projector import (
    EffectiveCapabilityHealthProjector,
)
from intergrax.applications._shared.capability_health.redaction import (
    safe_effective_capability_health_view,
)
from intergrax.applications._shared.profile_resolution.diff_engine import (
    diff_effective_profile_revisions,
)
from intergrax.applications._shared.runtime_inspection.composition import (
    default_runtime_inspection_providers,
)
from intergrax.applications._shared.runtime_inspection.merge import (
    invoke_provider_safely,
    merge_provider_contributions,
)
from intergrax.applications._shared.runtime_inspection.redaction import (
    safe_effective_profile_diff_view,
    safe_effective_profile_revision_view,
    safe_profile_resolution_view,
    sanitize_extension_evidence,
)
from intergrax.applications.contracts.capability_health import (
    CapabilityHealthProjectionContext,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
    EffectiveProfileExecutionPinningStore,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.applications.contracts.runtime_inspection.completeness import (
    InspectionCompleteness,
)
from intergrax.applications.contracts.runtime_inspection.inconsistency import (
    InspectionInconsistency,
    InspectionInconsistencyKind,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.runtime_inspection.results import (
    CapabilityInspectionResult,
    ExecutionInspectionResult,
    ProfileInspectionResult,
    RevisionCompareResult,
    RevisionInspectionResult,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.skills.registry.runtime import SkillRegistry


def _sorted_providers(
    providers: Sequence[RuntimeInspectionProvider],
) -> tuple[RuntimeInspectionProvider, ...]:
    return tuple(sorted(providers, key=lambda item: item.provider_id))


def _execution_inconsistencies(
    *,
    binding: EffectiveProfileExecutionBinding | None,
    pinned_revision: EffectiveProfileRevision | None,
    scope_application_id: str,
    scope_tenant_id: str | None,
) -> tuple[InspectionInconsistency, ...]:
    inconsistencies: list[InspectionInconsistency] = []
    if binding is None:
        return (
            InspectionInconsistency(
                kind=InspectionInconsistencyKind.NOT_FOUND,
                message="execution revision binding not found",
            ),
        )
    if pinned_revision is None:
        inconsistencies.append(
            InspectionInconsistency(
                kind=InspectionInconsistencyKind.MISSING_REVISION,
                message=f"pinned revision {binding.revision_id.value} is unavailable",
                field="revision_id",
            ),
        )
        return tuple(inconsistencies)
    if pinned_revision.fingerprint != binding.fingerprint:
        inconsistencies.append(
            InspectionInconsistency(
                kind=InspectionInconsistencyKind.FINGERPRINT_MISMATCH,
                message="binding fingerprint does not match revision fingerprint",
                field="fingerprint",
            ),
        )
    if pinned_revision.scope.application_id != scope_application_id:
        inconsistencies.append(
            InspectionInconsistency(
                kind=InspectionInconsistencyKind.APPLICATION_SCOPE_MISMATCH,
                message="pinned revision application scope mismatch",
                field="scope.application_id",
            ),
        )
    if scope_tenant_id is not None and pinned_revision.scope.tenant_id != scope_tenant_id:
        inconsistencies.append(
            InspectionInconsistency(
                kind=InspectionInconsistencyKind.TENANT_SCOPE_MISMATCH,
                message="pinned revision tenant scope mismatch",
                field="scope.tenant_id",
            ),
        )
    return tuple(inconsistencies)


class RuntimeInspectionService:
    """Read-only cross-domain inspection aggregator — never runtime authority."""

    def __init__(
        self,
        *,
        revision_store: EffectiveProfileRevisionStore | None = None,
        pinning_store: EffectiveProfileExecutionPinningStore | None = None,
        providers: Sequence[RuntimeInspectionProvider] | None = None,
        health_projector: EffectiveCapabilityHealthProjector | None = None,
    ) -> None:
        self._revision_store = revision_store
        self._pinning_store = pinning_store
        self._providers = _sorted_providers(
            providers if providers is not None else default_runtime_inspection_providers(),
        )
        self._health_projector = health_projector or EffectiveCapabilityHealthProjector(
            default_capability_health_providers(),
        )

    @property
    def providers(self) -> tuple[RuntimeInspectionProvider, ...]:
        return self._providers

    def inspect_profile(
        self,
        resolution: ProfileResolution,
        *,
        configured_profile_ref: str | None = None,
    ) -> ProfileInspectionResult:
        contributions = tuple(
            invoke_provider_safely(
                provider,
                method_name="contribute_profile",
                resolution=resolution,
                configured_profile_ref=configured_profile_ref,
            )
            for provider in self._providers
        )
        completeness, explanations, extensions, failures = merge_provider_contributions(
            contributions,
        )
        return ProfileInspectionResult(
            configured_profile_ref=configured_profile_ref,
            resolution=resolution,
            safe_resolution=safe_profile_resolution_view(resolution),
            completeness=completeness,
            explanations=explanations,
            provider_failures=failures,
            extension_evidence=tuple(sanitize_extension_evidence(item) for item in extensions),
        )

    def inspect_revision(
        self,
        revision_id: EffectiveProfileRevisionId,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> RevisionInspectionResult:
        revision = None
        inconsistencies: list[InspectionInconsistency] = []
        if self._revision_store is None:
            inconsistencies.append(
                InspectionInconsistency(
                    kind=InspectionInconsistencyKind.INCOMPLETE,
                    message="revision store is not configured",
                ),
            )
            completeness = InspectionCompleteness.UNAVAILABLE
        else:
            revision = self._revision_store.get(revision_id, scope=scope)
            if revision is None:
                inconsistencies.append(
                    InspectionInconsistency(
                        kind=InspectionInconsistencyKind.NOT_FOUND,
                        message=f"revision {revision_id.value} not found",
                        field="revision_id",
                    ),
                )
            completeness = (
                InspectionCompleteness.COMPLETE
                if revision is not None
                else InspectionCompleteness.UNAVAILABLE
            )
        contributions = tuple(
            invoke_provider_safely(
                provider,
                method_name="contribute_revision",
                revision_id=revision_id,
                scope=scope,
                revision=revision,
            )
            for provider in self._providers
        )
        provider_completeness, explanations, extensions, failures = merge_provider_contributions(
            contributions,
        )
        if provider_completeness is InspectionCompleteness.PARTIAL and completeness is (
            InspectionCompleteness.COMPLETE
        ):
            completeness = InspectionCompleteness.PARTIAL
        return RevisionInspectionResult(
            revision_id=revision_id,
            scope=scope,
            revision=revision,
            safe_revision=(
                safe_effective_profile_revision_view(revision) if revision is not None else None
            ),
            completeness=completeness,
            inconsistencies=tuple(inconsistencies),
            explanations=explanations,
            provider_failures=failures,
            extension_evidence=tuple(sanitize_extension_evidence(item) for item in extensions),
        )

    def compare_revisions(
        self,
        from_revision: EffectiveProfileRevision,
        to_revision: EffectiveProfileRevision,
    ) -> RevisionCompareResult:
        diff = diff_effective_profile_revisions(from_revision, to_revision)
        contributions = tuple(
            invoke_provider_safely(
                provider,
                method_name="contribute_revision_compare",
                from_revision=from_revision,
                to_revision=to_revision,
            )
            for provider in self._providers
        )
        completeness, _, extensions, failures = merge_provider_contributions(contributions)
        return RevisionCompareResult(
            diff=diff,
            safe_diff=safe_effective_profile_diff_view(diff),
            completeness=completeness,
            provider_failures=failures,
            extension_evidence=tuple(sanitize_extension_evidence(item) for item in extensions),
        )

    def inspect_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        scope_application_id: str,
        scope_tenant_id: str | None = None,
    ) -> ExecutionInspectionResult:
        binding: EffectiveProfileExecutionBinding | None = None
        pinned_revision: EffectiveProfileRevision | None = None
        if self._pinning_store is None:
            inconsistencies = (
                InspectionInconsistency(
                    kind=InspectionInconsistencyKind.INCOMPLETE,
                    message="execution pinning store is not configured",
                ),
            )
            completeness = InspectionCompleteness.UNAVAILABLE
        else:
            binding = self._pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
            if binding is not None and self._revision_store is not None:
                pinned_revision = self._revision_store.get(
                    binding.revision_id,
                    scope=EffectiveProfileRevisionScope(
                        application_id=scope_application_id,
                        tenant_id=scope_tenant_id,
                    ),
                )
            inconsistencies = list(
                _execution_inconsistencies(
                    binding=binding,
                    pinned_revision=pinned_revision,
                    scope_application_id=scope_application_id,
                    scope_tenant_id=scope_tenant_id,
                ),
            )
            if binding is None:
                completeness = InspectionCompleteness.UNAVAILABLE
            elif inconsistencies:
                completeness = InspectionCompleteness.PARTIAL
            else:
                completeness = InspectionCompleteness.COMPLETE
        contributions = tuple(
            invoke_provider_safely(
                provider,
                method_name="contribute_execution",
                tenant_id=tenant_id,
                execution_id=execution_id,
                scope_application_id=scope_application_id,
                scope_tenant_id=scope_tenant_id,
                binding=binding,
                pinned_revision=pinned_revision,
            )
            for provider in self._providers
        )
        provider_completeness, explanations, extensions, failures = merge_provider_contributions(
            contributions,
        )
        if provider_completeness is InspectionCompleteness.PARTIAL and completeness in {
            InspectionCompleteness.COMPLETE,
            InspectionCompleteness.PARTIAL,
        }:
            completeness = InspectionCompleteness.PARTIAL
        return ExecutionInspectionResult(
            tenant_id=tenant_id,
            execution_id=execution_id,
            scope_application_id=scope_application_id,
            scope_tenant_id=scope_tenant_id,
            binding=binding,
            pinned_revision=pinned_revision,
            safe_pinned_revision=(
                safe_effective_profile_revision_view(pinned_revision)
                if pinned_revision is not None
                else None
            ),
            completeness=completeness,
            inconsistencies=tuple(inconsistencies),
            explanations=explanations,
            provider_failures=failures,
            extension_evidence=tuple(sanitize_extension_evidence(item) for item in extensions),
        )

    def inspect_capability(
        self,
        capability: CapabilityRef,
        validation: CapabilityDependencyValidationResult,
        *,
        environment_profile: ApplicationEnvironmentProfile | None = None,
        scope_application_id: str | None = None,
        scope_tenant_id: str | None = None,
        effective_profile_revision_id: EffectiveProfileRevisionId | None = None,
        effective_profile_fingerprint: str | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> CapabilityInspectionResult:
        capability_key = capability.canonical_key
        outcome = next(
            (item for item in validation.outcomes if item.owner.canonical_key == capability_key),
            None,
        )
        required_failures = tuple(
            sorted(
                (
                    failure
                    for failure in validation.required_failures
                    if failure.owner.canonical_key == capability_key
                ),
                key=lambda item: (
                    item.dependency.canonical_key,
                    item.requirement.value,
                    item.status.value,
                ),
            ),
        )
        optional_degradations = tuple(
            sorted(
                (
                    degradation
                    for degradation in validation.optional_degradations
                    if degradation.owner.canonical_key == capability_key
                ),
                key=lambda item: (
                    item.dependency.canonical_key,
                    item.requirement.value,
                    item.status.value,
                ),
            ),
        )
        contributions = tuple(
            invoke_provider_safely(
                provider,
                method_name="contribute_capability",
                capability_key=capability_key,
                validation=validation,
            )
            for provider in self._providers
        )
        completeness, explanations, extensions, failures = merge_provider_contributions(
            contributions,
        )
        health = self._health_projector.project(
            CapabilityHealthProjectionContext(
                capability=capability,
                validation=validation,
                environment_profile=environment_profile,
                scope_application_id=scope_application_id,
                scope_tenant_id=scope_tenant_id,
                effective_profile_revision_id=effective_profile_revision_id,
                effective_profile_fingerprint=effective_profile_fingerprint,
                skill_registry=skill_registry,
            ),
        )
        return CapabilityInspectionResult(
            capability=capability,
            validation=validation,
            outcome=outcome,
            required_failures=required_failures,
            optional_degradations=optional_degradations,
            health=health,
            safe_health=safe_effective_capability_health_view(health),
            completeness=completeness,
            explanations=explanations,
            provider_failures=failures,
            extension_evidence=tuple(sanitize_extension_evidence(item) for item in extensions),
        )
