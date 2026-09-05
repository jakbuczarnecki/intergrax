# © Artur Czarnecki. All rights reserved.

"""Composition and profile-resolution adoption for capability dependency validation (P1.3)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications._shared.capability_dependency.skill_tool_provider import (
    SkillToolCapabilityDependencyProvider,
)
from intergrax.applications._shared.capability_dependency.validator import (
    CapabilityDependencyValidator,
)
from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependencyProvider,
    CapabilityDependencyValidationContext,
    CapabilityDependencyValidationResult,
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    DegradedCapability,
    ProfileDependencyFailure,
    ProfileResolution,
)
from intergrax.skills.registry.runtime import SkillRegistry


def default_capability_dependency_providers() -> tuple[CapabilityDependencyProvider, ...]:
    """Assembled provider projection — no global hard-coded dependency registry."""
    return (SkillToolCapabilityDependencyProvider(),)


def validate_capability_dependencies(
    context: CapabilityDependencyValidationContext,
    *,
    providers: Sequence[CapabilityDependencyProvider] | None = None,
) -> CapabilityDependencyValidationResult:
    """Validate declared dependencies for one environment composition context."""
    resolved_providers = (
        tuple(providers)
        if providers is not None
        else default_capability_dependency_providers()
    )
    return CapabilityDependencyValidator(resolved_providers).validate(context)


def validate_capability_dependencies_for_environment(
    environment_profile: ApplicationEnvironmentProfile,
    *,
    skill_registry: SkillRegistry | None = None,
    providers: Sequence[CapabilityDependencyProvider] | None = None,
    fail_closed: bool = True,
) -> CapabilityDependencyValidationResult:
    """Composition-time gate: required missing dependencies block host wiring."""
    context = CapabilityDependencyValidationContext(
        environment_profile=environment_profile,
        skill_registry=skill_registry,
    )
    result = validate_capability_dependencies(context, providers=providers)
    if fail_closed and result.required_failures:
        raise RequiredCapabilityDependencyUnavailableError(result)
    return result


def map_validation_to_profile_resolution_evidence(
    result: CapabilityDependencyValidationResult,
) -> tuple[tuple[ProfileDependencyFailure, ...], tuple[DegradedCapability, ...]]:
    """Project dependency validation facts into P1.1 evidence carriers."""
    failures = tuple(
        ProfileDependencyFailure(
            capability=failure.owner.canonical_key,
            reason=failure.reason,
            dependency_kind=failure.dependency_kind.value,
            dependency_id=failure.dependency.capability_id,
            requirement=failure.requirement.value,
            status=failure.status.value,
            source_domain=failure.source_domain,
        )
        for failure in result.required_failures
    )
    degraded = tuple(
        DegradedCapability(
            capability=degradation.owner.canonical_key,
            reason=degradation.reason,
            dependency_kind=degradation.dependency_kind.value,
            dependency_id=degradation.dependency.capability_id,
            requirement=degradation.requirement.value,
            status=degradation.status.value,
            source_domain=degradation.source_domain,
        )
        for degradation in result.optional_degradations
    )
    return failures, degraded


def enrich_profile_resolution_with_capability_dependencies(
    resolution: ProfileResolution,
    *,
    skill_registry: SkillRegistry | None = None,
    providers: Sequence[CapabilityDependencyProvider] | None = None,
) -> ProfileResolution:
    """Attach dependency validation evidence to one ProfileResolution read model."""
    context = CapabilityDependencyValidationContext(
        environment_profile=resolution.effective_profile,
        skill_registry=skill_registry,
    )
    validation = validate_capability_dependencies(context, providers=providers)
    dependency_failures, degraded_capabilities = map_validation_to_profile_resolution_evidence(
        validation,
    )
    return resolution.model_copy(
        update={
            "dependency_failures": dependency_failures,
            "degraded_capabilities": degraded_capabilities,
        },
    )
