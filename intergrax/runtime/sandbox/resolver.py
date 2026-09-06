# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure effective execution environment resolver (P1.8)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.applications.contracts.profile_resolution.revision import EffectiveProfileRevision
from intergrax.runtime.sandbox.execution_environment import (
    EffectiveExecutionEnvironment,
    ExecutionEnvironmentProvenance,
    ExecutionEnvironmentProviderKind,
    ExecutionEnvironmentProviderRef,
    ExecutionEnvironmentRequirement,
    ExecutionEnvironmentResolutionFailure,
    ExecutionEnvironmentResolutionFailureReason,
    ExecutionEnvironmentResolutionResult,
    FilesystemAccess,
    NetworkAccess,
    PrivilegeMode,
    ProcessExecution,
    ProfileIsolationAuthority,
    SandboxProviderCapabilities,
)


_FILESYSTEM_ORDER: tuple[FilesystemAccess, ...] = (
    FilesystemAccess.NONE,
    FilesystemAccess.READ_ONLY,
    FilesystemAccess.WORKSPACE_WRITE,
)
_NETWORK_ORDER: tuple[NetworkAccess, ...] = (
    NetworkAccess.NONE,
    NetworkAccess.RESTRICTED,
    NetworkAccess.ALLOWED,
)
_PROCESS_ORDER: tuple[ProcessExecution, ...] = (
    ProcessExecution.DENIED,
    ProcessExecution.SANDBOXED,
)
_PRIVILEGE_ORDER: tuple[PrivilegeMode, ...] = (
    PrivilegeMode.STANDARD,
    PrivilegeMode.PRIVILEGED,
)


def _index(order: tuple[object, ...], value: object) -> int:
    return order.index(value)


def _at_most_as_permissive[T](order: tuple[T, ...], ceiling: T, value: T) -> T:
    return order[min(_index(order, ceiling), _index(order, value))]


def _meets_minimum(order: tuple[object, ...], effective: object, required: object) -> bool:
    return _index(order, effective) >= _index(order, required)


def profile_isolation_authority(profile: ApplicationEnvironmentProfile) -> ProfileIsolationAuthority:
    """Derive isolation ceiling from configured profile — narrowing-only authority."""
    sandbox: SandboxProfile | None = profile.sandbox
    if sandbox is None:
        return ProfileIsolationAuthority()
    # SandboxProfile contract: session manager root/session configuration only.
    # Configured sandbox isolation permits sandboxed process execution and workspace-scoped
    # filesystem within the session root; it does not declare network isolation policy.
    return ProfileIsolationAuthority(
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=NetworkAccess.NONE,
        process_execution=ProcessExecution.SANDBOXED,
        privilege_mode=PrivilegeMode.STANDARD,
        sandbox_configured=True,
    )


def validate_child_requirement_not_widening(
    parent: ExecutionEnvironmentRequirement,
    child: ExecutionEnvironmentRequirement,
) -> ExecutionEnvironmentResolutionFailure | None:
    """Child authority must not exceed parent environment authority."""
    if child.sandbox_required and not parent.sandbox_required:
        return ExecutionEnvironmentResolutionFailure(
            reason=ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION,
            message="child sandbox requirement exceeds parent authority",
            reason_codes=("child.widens.sandbox",),
        )
    for order, field in (
        (_FILESYSTEM_ORDER, "filesystem_access"),
        (_NETWORK_ORDER, "network_access"),
        (_PROCESS_ORDER, "process_execution"),
        (_PRIVILEGE_ORDER, "privilege_mode"),
    ):
        parent_value = getattr(parent, field)
        child_value = getattr(child, field)
        if _index(order, child_value) > _index(order, parent_value):
            return ExecutionEnvironmentResolutionFailure(
                reason=ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION,
                message=f"child {field} exceeds parent authority",
                reason_codes=("child.widens.authority", field),
            )
    return None


def _authority_violation(
    *,
    field: str,
    profile: ProfileIsolationAuthority,
    requirement: ExecutionEnvironmentRequirement,
) -> ExecutionEnvironmentResolutionResult:
    return ExecutionEnvironmentResolutionResult(
        failure=ExecutionEnvironmentResolutionFailure(
            reason=ExecutionEnvironmentResolutionFailureReason.AUTHORITY_VIOLATION,
            message=f"requirement {field} exceeds profile isolation authority",
            reason_codes=("profile.authority.clamped", field),
        ),
    )


def _provider_satisfies(
    provider: SandboxProviderCapabilities,
    requirement: ExecutionEnvironmentRequirement,
) -> bool:
    if requirement.sandbox_required and provider.provider_ref.provider_kind is ExecutionEnvironmentProviderKind.NONE:
        return False
    if requirement.process_execution is ProcessExecution.SANDBOXED and not provider.supports_sandboxed_exec:
        return False
    if (
        requirement.filesystem_access is FilesystemAccess.WORKSPACE_WRITE
        and not provider.supports_workspace_write
    ):
        return False
    if not _meets_minimum(_FILESYSTEM_ORDER, provider.filesystem_access, requirement.filesystem_access):
        return False
    if not _meets_minimum(_NETWORK_ORDER, provider.network_access, requirement.network_access):
        return False
    if not _meets_minimum(_PROCESS_ORDER, provider.process_execution, requirement.process_execution):
        return False
    return True


def resolve_effective_execution_environment(
    *,
    profile_authority: ProfileIsolationAuthority,
    requirement: ExecutionEnvironmentRequirement,
    provider_capabilities: SandboxProviderCapabilities | None,
) -> ExecutionEnvironmentResolutionResult:
    """
    Pure resolution: requirement ∩ profile authority ∩ provider capabilities.

    Does not create sessions, probe networks, or mutate profile state.
    """
    if requirement.sandbox_required and not profile_authority.sandbox_configured:
        return _authority_violation(
            field="sandbox_required",
            profile=profile_authority,
            requirement=requirement,
        )

    for order, field in (
        (_FILESYSTEM_ORDER, "filesystem_access"),
        (_NETWORK_ORDER, "network_access"),
        (_PROCESS_ORDER, "process_execution"),
        (_PRIVILEGE_ORDER, "privilege_mode"),
    ):
        required = getattr(requirement, field)
        allowed = getattr(profile_authority, field)
        if _index(order, required) > _index(order, allowed):
            return _authority_violation(field=field, profile=profile_authority, requirement=requirement)

    if requirement.sandbox_required and provider_capabilities is None:
        return ExecutionEnvironmentResolutionResult(
            failure=ExecutionEnvironmentResolutionFailure(
                reason=ExecutionEnvironmentResolutionFailureReason.PROVIDER_UNAVAILABLE,
                message="sandbox required but no provider capabilities available",
                reason_codes=("provider.unavailable",),
            ),
        )

    if provider_capabilities is None:
        provider_ref = ExecutionEnvironmentProviderRef(
            provider_id="none",
            provider_kind=ExecutionEnvironmentProviderKind.NONE,
        )
        effective = EffectiveExecutionEnvironment(
            filesystem_access=FilesystemAccess.NONE,
            network_access=NetworkAccess.NONE,
            process_execution=ProcessExecution.DENIED,
            privilege_mode=PrivilegeMode.STANDARD,
            sandbox_required=False,
            provider_ref=provider_ref,
            provenance=ExecutionEnvironmentProvenance(
                profile_contribution=str(profile_authority.model_dump(mode="json")),
                requirement_contribution=str(requirement.model_dump(mode="json")),
                provider_contribution="none",
                decision="non_sandbox_environment",
                reason_codes=("sandbox.optional",),
            ),
        )
        if requirement.sandbox_required:
            return ExecutionEnvironmentResolutionResult(
                failure=ExecutionEnvironmentResolutionFailure(
                    reason=ExecutionEnvironmentResolutionFailureReason.REQUIREMENT_UNSATISFIED,
                    message="sandbox required but resolved non-sandbox environment",
                    reason_codes=("requirement.unsatisfied",),
                ),
            )
        return ExecutionEnvironmentResolutionResult(environment=effective)

    if not _provider_satisfies(provider_capabilities, requirement):
        return ExecutionEnvironmentResolutionResult(
            failure=ExecutionEnvironmentResolutionFailure(
                reason=ExecutionEnvironmentResolutionFailureReason.PROVIDER_CAPABILITY_UNSATISFIED,
                message="provider lacks required execution environment capability",
                reason_codes=("provider.unsatisfied",),
            ),
        )

    effective = EffectiveExecutionEnvironment(
        filesystem_access=_at_most_as_permissive(
            _FILESYSTEM_ORDER,
            profile_authority.filesystem_access,
            _at_most_as_permissive(
                _FILESYSTEM_ORDER,
                provider_capabilities.filesystem_access,
                requirement.filesystem_access,
            ),
        ),
        network_access=_at_most_as_permissive(
            _NETWORK_ORDER,
            profile_authority.network_access,
            _at_most_as_permissive(
                _NETWORK_ORDER,
                provider_capabilities.network_access,
                requirement.network_access,
            ),
        ),
        process_execution=_at_most_as_permissive(
            _PROCESS_ORDER,
            profile_authority.process_execution,
            _at_most_as_permissive(
                _PROCESS_ORDER,
                provider_capabilities.process_execution,
                requirement.process_execution,
            ),
        ),
        privilege_mode=_at_most_as_permissive(
            _PRIVILEGE_ORDER,
            profile_authority.privilege_mode,
            requirement.privilege_mode,
        ),
        sandbox_required=requirement.sandbox_required,
        provider_ref=provider_capabilities.provider_ref,
        provenance=ExecutionEnvironmentProvenance(
            profile_contribution=str(profile_authority.model_dump(mode="json")),
            requirement_contribution=str(requirement.model_dump(mode="json")),
            provider_contribution=str(provider_capabilities.model_dump(mode="json")),
            decision="narrowed_effective_environment",
            reason_codes=(
                "profile.allowed",
                "tool.requires" if requirement.sandbox_required else "tool.optional",
                "provider.supports",
            ),
        ),
    )

    return ExecutionEnvironmentResolutionResult(environment=effective)


def resolve_effective_execution_environment_for_profile(
    profile: ApplicationEnvironmentProfile,
    requirement: ExecutionEnvironmentRequirement,
    provider_capabilities: SandboxProviderCapabilities | None,
) -> ExecutionEnvironmentResolutionResult:
    return resolve_effective_execution_environment(
        profile_authority=profile_isolation_authority(profile),
        requirement=requirement,
        provider_capabilities=provider_capabilities,
    )


def resolve_effective_execution_environment_for_revision(
    revision: EffectiveProfileRevision,
    requirement: ExecutionEnvironmentRequirement,
    provider_capabilities: SandboxProviderCapabilities | None,
) -> ExecutionEnvironmentResolutionResult:
    """Deterministic environment from pinned effective profile revision."""
    return resolve_effective_execution_environment_for_profile(
        revision.effective_profile,
        requirement,
        provider_capabilities,
    )
