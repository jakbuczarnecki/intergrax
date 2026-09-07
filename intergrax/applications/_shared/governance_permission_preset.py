# © Artur Czarnecki. All rights reserved.

"""Monotonic governance permission preset expansion (P1.11)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CapabilityBundle,
    DomainPolicyFragment,
    DomainPolicyFragments,
    GovernanceBundle,
    IsolationBundle,
)
from intergrax.applications.contracts.environment_profile.governance_permission_preset import (
    GovernancePermissionPreset,
    GovernancePermissionPresetProvenance,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ContextProfile,
    PromptProfile,
    ReliabilityProfile,
    SandboxProfile,
)
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.tools.registry.profile import ToolProfile

_GOVERNANCE_PRESET_FRAGMENT_ID = "governance.permission_preset"

_AUTONOMY_RESTRICTION_ORDER: dict[AutonomyLevel, int] = {
    AutonomyLevel.MANUAL: 0,
    AutonomyLevel.ASK: 1,
    AutonomyLevel.AUTONOMOUS: 2,
}


def _intersect_tool_profiles(
    *,
    upstream: ToolProfile,
    requested: ToolProfile,
) -> ToolProfile:
    from intergrax.applications._shared.profile_resolution.field_resolvers import (
        intersect_tool_profiles,
    )

    return intersect_tool_profiles(
        upstream=upstream,
        requested=requested,
        upstream_expressed=True,
    )


def preset_opinion_profile(preset: GovernancePermissionPreset) -> ApplicationEnvironmentProfile:
    """Return canonical profile opinion for a preset — not an authority surface."""
    if preset is GovernancePermissionPreset.BALANCED:
        return ApplicationEnvironmentProfile()
    if preset is GovernancePermissionPreset.RESTRICTED:
        return ApplicationEnvironmentProfile(
            capabilities=CapabilityBundle(
                prompt=PromptProfile(approval_required=True),
                context=ContextProfile(enable_websearch=False),
            ),
            governance=GovernanceBundle(
                reliability=ReliabilityProfile(
                    default_autonomy_level=AutonomyLevel.MANUAL,
                    tenant_autonomy_ceiling=AutonomyLevel.MANUAL,
                ),
            ),
            isolation=IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=False),
            ),
        )
    return ApplicationEnvironmentProfile(
        capabilities=CapabilityBundle(
            prompt=PromptProfile(approval_required=False),
            context=ContextProfile(enable_websearch=True),
            tools=ToolProfile(
                enabled=["read_file", "write_file", "sandbox.exec"],
            ),
        ),
        governance=GovernanceBundle(
            reliability=ReliabilityProfile(
                default_autonomy_level=AutonomyLevel.AUTONOMOUS,
            ),
        ),
        isolation=IsolationBundle(
            sandbox=SandboxProfile(enable_exec_tool=True),
        ),
    )


def resolve_require_human_on_critical(env: ApplicationEnvironmentProfile) -> bool:
    """Derive HITL posture from canonical profile fields (fail-closed default)."""
    if env.prompt_profile.approval_required:
        return True
    if env.reliability_profile.default_autonomy_level is not AutonomyLevel.AUTONOMOUS:
        return True
    if env.compliance_profile.enabled:
        return True
    preset = env.governance.permission_preset
    if preset is GovernancePermissionPreset.RESTRICTED:
        return True
    return True


def _more_restrictive_autonomy(
    configured: AutonomyLevel,
    preset_opinion: AutonomyLevel,
) -> AutonomyLevel:
    if (
        _AUTONOMY_RESTRICTION_ORDER[configured]
        <= _AUTONOMY_RESTRICTION_ORDER[preset_opinion]
    ):
        return configured
    return preset_opinion


def _more_restrictive_autonomy_ceiling(
    configured: AutonomyLevel | None,
    preset_opinion: AutonomyLevel | None,
) -> AutonomyLevel | None:
    if preset_opinion is None:
        return configured
    if configured is None:
        return preset_opinion
    if (
        _AUTONOMY_RESTRICTION_ORDER[configured]
        <= _AUTONOMY_RESTRICTION_ORDER[preset_opinion]
    ):
        return configured
    return preset_opinion


def _more_restrictive_bool(
    configured: bool,
    preset_opinion: bool,
    *,
    restrictive_when: bool,
) -> bool:
    configured_rank = 0 if configured == restrictive_when else 1
    preset_rank = 0 if preset_opinion == restrictive_when else 1
    return configured if configured_rank <= preset_rank else preset_opinion


def _merge_sandbox_profile(
    configured: SandboxProfile | None,
    preset_opinion: SandboxProfile | None,
) -> SandboxProfile | None:
    if configured is None:
        return None
    if preset_opinion is None:
        return configured
    merged_exec = _more_restrictive_bool(
        configured.enable_exec_tool,
        preset_opinion.enable_exec_tool,
        restrictive_when=False,
    )
    if merged_exec == configured.enable_exec_tool:
        return configured
    return configured.model_copy(update={"enable_exec_tool": merged_exec})


def _collect_clamped_fields(
    configured: ApplicationEnvironmentProfile,
    opinion: ApplicationEnvironmentProfile,
    merged: ApplicationEnvironmentProfile,
) -> tuple[str, ...]:
    clamped: list[str] = []
    if merged.prompt_profile.approval_required != opinion.prompt_profile.approval_required:
        clamped.append("capabilities.prompt.approval_required")
    if merged.context_profile.enable_websearch != opinion.context_profile.enable_websearch:
        clamped.append("capabilities.context.enable_websearch")
    if (
        merged.reliability_profile.default_autonomy_level
        != opinion.reliability_profile.default_autonomy_level
    ):
        clamped.append("governance.reliability.default_autonomy_level")
    if merged.tool_profile != opinion.tool_profile:
        clamped.append("capabilities.tools")
    configured_sandbox = configured.sandbox
    opinion_sandbox = opinion.sandbox
    merged_sandbox = merged.sandbox
    if (
        opinion_sandbox is not None
        and configured_sandbox is not None
        and merged_sandbox is not None
        and merged_sandbox.enable_exec_tool != opinion_sandbox.enable_exec_tool
    ):
        clamped.append("isolation.sandbox.enable_exec_tool")
    return tuple(clamped)


def _attach_preset_provenance(
    profile: ApplicationEnvironmentProfile,
    provenance: GovernancePermissionPresetProvenance,
) -> ApplicationEnvironmentProfile:
    existing = profile.extensions.domain_policy_fragments
    fragment = DomainPolicyFragment(
        fragment_id=_GOVERNANCE_PRESET_FRAGMENT_ID,
        payload={
            "requested_preset": provenance.requested_preset.value,
            "effective_require_human_on_critical": provenance.effective_require_human_on_critical,
            "clamped_fields": list(provenance.clamped_fields),
        },
    )
    merged_fragments = DomainPolicyFragments(
        fragments={**existing.fragments, _GOVERNANCE_PRESET_FRAGMENT_ID: fragment},
    )
    return profile.model_copy(
        update={
            "extensions": profile.extensions.model_copy(
                update={"domain_policy_fragments": merged_fragments},
            ),
        },
    )


def expand_governance_permission_preset(
    profile: ApplicationEnvironmentProfile,
) -> ApplicationEnvironmentProfile:
    """
    Expand configured preset into canonical profile fields before resolution.

    Preset expansion may narrow but never widen upstream configured authority.
    """
    preset = profile.governance.permission_preset
    if preset is None:
        return profile

    if preset is GovernancePermissionPreset.BALANCED:
        provenance = GovernancePermissionPresetProvenance(
            requested_preset=preset,
            effective_require_human_on_critical=resolve_require_human_on_critical(profile),
            clamped_fields=(),
        )
        return _attach_preset_provenance(
            profile.model_copy(
                update={
                    "governance": profile.governance.model_copy(
                        update={"permission_preset": preset},
                    ),
                },
            ),
            provenance,
        )

    opinion = preset_opinion_profile(preset)
    merged_prompt = profile.prompt_profile.model_copy(
        update={
            "approval_required": _more_restrictive_bool(
                profile.prompt_profile.approval_required,
                opinion.prompt_profile.approval_required,
                restrictive_when=True,
            ),
        },
    )
    merged_context = profile.context_profile.model_copy(
        update={
            "enable_websearch": _more_restrictive_bool(
                profile.context_profile.enable_websearch,
                opinion.context_profile.enable_websearch,
                restrictive_when=False,
            ),
        },
    )
    merged_reliability = profile.reliability_profile.model_copy(
        update={
            "default_autonomy_level": _more_restrictive_autonomy(
                profile.reliability_profile.default_autonomy_level,
                opinion.reliability_profile.default_autonomy_level,
            ),
            "tenant_autonomy_ceiling": _more_restrictive_autonomy_ceiling(
                profile.reliability_profile.tenant_autonomy_ceiling,
                opinion.reliability_profile.tenant_autonomy_ceiling,
            ),
        },
    )
    merged_tools = profile.tool_profile
    if preset is GovernancePermissionPreset.TRUSTED and (
        opinion.tool_profile.enabled
        or opinion.tool_profile.enabled_bundles
        or opinion.tool_profile.register_all_catalog_bundles
    ):
        merged_tools = _intersect_tool_profiles(
            upstream=profile.tool_profile,
            requested=opinion.tool_profile,
        )
    merged_sandbox = _merge_sandbox_profile(profile.sandbox, opinion.sandbox)

    merged = profile.model_copy(
        update={
            "capabilities": profile.capabilities.model_copy(
                update={
                    "prompt": merged_prompt,
                    "context": merged_context,
                    "tools": merged_tools,
                },
            ),
            "governance": profile.governance.model_copy(
                update={
                    "reliability": merged_reliability,
                    "permission_preset": preset,
                },
            ),
            "isolation": profile.isolation.model_copy(
                update={"sandbox": merged_sandbox},
            ),
        },
    )
    provenance = GovernancePermissionPresetProvenance(
        requested_preset=preset,
        effective_require_human_on_critical=resolve_require_human_on_critical(merged),
        clamped_fields=_collect_clamped_fields(profile, opinion, merged),
    )
    return _attach_preset_provenance(merged, provenance)
