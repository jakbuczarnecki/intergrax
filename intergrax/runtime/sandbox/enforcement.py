# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve and enforce effective execution environment before sandbox tool effects (P1.8)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution.revision import EffectiveProfileRevision
from intergrax.runtime.sandbox.execution_environment import (
    ExecutionEnvironmentRequirement,
    ExecutionEnvironmentResolutionFailureReason,
    FilesystemAccess,
    NetworkAccess,
    ProcessExecution,
    PrivilegeMode,
    ProfileIsolationAuthority,
)
from intergrax.runtime.sandbox.provider_adapters import (
    probe_provider_capabilities_from_wiring,
    select_provider_capabilities,
)
from intergrax.runtime.sandbox.resolver import (
    resolve_effective_execution_environment,
    resolve_effective_execution_environment_for_profile,
    resolve_effective_execution_environment_for_revision,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.providers.sandbox.contracts import SandboxExecOutput
from intergrax.tools.registry.wiring import ToolWiringContext


def _failure_error_code(failure_reason: ExecutionEnvironmentResolutionFailureReason) -> str:
    return f"execution_environment_{failure_reason.value}"


def _profile_from_context(ctx: ToolWiringContext) -> ApplicationEnvironmentProfile | None:
    raw = ctx.extras.get("effective_environment_profile")
    if isinstance(raw, ApplicationEnvironmentProfile):
        return raw
    revision_raw = ctx.extras.get("effective_profile_revision")
    if isinstance(revision_raw, EffectiveProfileRevision):
        return revision_raw.effective_profile
    return None


def resolve_tool_execution_environment(
    ctx: ToolWiringContext,
    *,
    requirement: ExecutionEnvironmentRequirement | None = None,
    contract: ToolContract | None = None,
) -> tuple[object | None, SandboxExecOutput | None]:
    """
    Resolve effective execution environment for a sandbox tool call.

    Returns ``(environment, None)`` on success or ``(None, error_output)`` on failure.
    """
    if requirement is None:
        if contract is None:
            requirement = ExecutionEnvironmentRequirement.none()
        else:
            requirement = ExecutionEnvironmentRequirement.from_tool_contract(contract)

    if not requirement.sandbox_required:
        return None, None

    profile = _profile_from_context(ctx)
    providers = probe_provider_capabilities_from_wiring(ctx)
    provider = select_provider_capabilities(providers)

    if profile is not None:
        result = resolve_effective_execution_environment_for_profile(
            profile,
            requirement,
            provider,
        )
    elif provider is None:
        return None, SandboxExecOutput(success=False, error="sandbox_session_not_configured")
    else:
        substrate_authority = ProfileIsolationAuthority(
            filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
            network_access=NetworkAccess.RESTRICTED,
            process_execution=ProcessExecution.SANDBOXED,
            privilege_mode=PrivilegeMode.STANDARD,
            sandbox_configured=True,
        )
        result = resolve_effective_execution_environment(
            profile_authority=substrate_authority,
            requirement=requirement,
            provider_capabilities=provider,
        )

    if result.failure is not None:
        return None, SandboxExecOutput(
            success=False,
            error=_failure_error_code(result.failure.reason),
        )
    return result.environment, None


def resolve_inspection_execution_environment(
    *,
    revision: EffectiveProfileRevision,
    requirement: ExecutionEnvironmentRequirement | None = None,
    provider_capabilities: object | None = None,
):
    """Read-only projection helper for runtime inspection."""
    from intergrax.runtime.sandbox.execution_environment import SandboxProviderCapabilities
    from intergrax.tools.core.contracts import ToolIsolationRequirement

    resolved_requirement = requirement or ExecutionEnvironmentRequirement.from_tool_isolation(
        ToolIsolationRequirement.SANDBOX,
    )
    provider = (
        provider_capabilities
        if isinstance(provider_capabilities, SandboxProviderCapabilities)
        else None
    )
    return resolve_effective_execution_environment_for_revision(
        revision,
        resolved_requirement,
        provider,
    )
