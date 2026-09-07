# © Artur Czarnecki. All rights reserved.

"""P1.11 — governance permission preset monotonic expansion proofs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications._shared.governance_permission_preset import (
    expand_governance_permission_preset,
    resolve_require_human_on_critical,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.profile_resolution import resolve_profile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    GovernanceBundle,
    IsolationBundle,
)
from intergrax.applications.contracts.environment_profile.governance_permission_preset import (
    GovernancePermissionPreset,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ContextProfile,
    PromptProfile,
    ReliabilityProfile,
    SandboxProfile,
)
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.sandbox.resolver import profile_isolation_authority
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _profile_with_preset(
    preset: GovernancePermissionPreset | None,
    *,
    tools: list[str] | None = None,
    approval_required: bool = False,
    enable_websearch: bool = True,
    enable_exec_tool: bool = True,
    autonomy: AutonomyLevel = AutonomyLevel.ASK,
) -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="preset.test")
    tool_profile = ToolProfile(enabled=tools or ["read_file"])
    return base.model_copy(
        update={
            "capabilities": base.capabilities.model_copy(
                update={
                    "tools": tool_profile,
                    "prompt": PromptProfile(approval_required=approval_required),
                    "context": ContextProfile(enable_websearch=enable_websearch),
                },
            ),
            "governance": GovernanceBundle(
                reliability=ReliabilityProfile(default_autonomy_level=autonomy),
                permission_preset=preset,
            ),
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=enable_exec_tool),
            ),
        },
    )


def test_preset_expands_into_canonical_runtime_policy() -> None:
    profile = _profile_with_preset(GovernancePermissionPreset.RESTRICTED)
    expanded = expand_governance_permission_preset(profile)
    bundle = wire_policy_bundle(expanded)

    assert expanded.prompt_profile.approval_required is True
    assert bundle.require_human_on_critical is True
    assert bundle.declarative_policy_runtime is None or bundle.policy_catalog is not None
    fragment = expanded.domain_policy_fragments["governance.permission_preset"]
    assert fragment["requested_preset"] == GovernancePermissionPreset.RESTRICTED.value


def test_preset_cannot_widen_tool_authority() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.TRUSTED,
        tools=["read_file"],
    )
    expanded = expand_governance_permission_preset(profile)

    assert set(expanded.tool_profile.enabled) == {"read_file"}
    assert expanded.tool_profile.register_all_catalog_bundles is False


def test_preset_cannot_widen_sandbox_authority() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.TRUSTED,
        enable_exec_tool=False,
    )
    expanded = expand_governance_permission_preset(profile)

    assert expanded.sandbox is not None
    assert expanded.sandbox.enable_exec_tool is False
    authority = profile_isolation_authority(expanded)
    assert authority.process_execution.value in {"denied", "sandboxed"}


def test_human_approval_restriction_is_monotonic() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.TRUSTED,
        approval_required=True,
        autonomy=AutonomyLevel.MANUAL,
    )
    expanded = expand_governance_permission_preset(profile)

    assert expanded.prompt_profile.approval_required is True
    assert expanded.reliability_profile.default_autonomy_level is AutonomyLevel.MANUAL
    assert resolve_require_human_on_critical(expanded) is True
    assert wire_policy_bundle(expanded).require_human_on_critical is True


def test_restrictive_preset_narrows_upstream_capability() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.RESTRICTED,
        enable_websearch=True,
        enable_exec_tool=True,
        autonomy=AutonomyLevel.ASK,
    )
    expanded = expand_governance_permission_preset(profile)

    assert expanded.context_profile.enable_websearch is False
    assert expanded.sandbox is not None
    assert expanded.sandbox.enable_exec_tool is False
    assert expanded.reliability_profile.default_autonomy_level is AutonomyLevel.MANUAL


def test_explicit_configuration_overrides_liberal_preset() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.TRUSTED,
        approval_required=True,
        tools=["read_file"],
        enable_exec_tool=False,
    )
    expanded = expand_governance_permission_preset(profile)
    fragment = expanded.domain_policy_fragments["governance.permission_preset"]

    assert expanded.prompt_profile.approval_required is True
    assert expanded.sandbox is not None
    assert expanded.sandbox.enable_exec_tool is False
    assert "capabilities.prompt.approval_required" in fragment["clamped_fields"]


def test_default_compatibility_without_preset() -> None:
    profile = _profile_with_preset(None)
    expanded = expand_governance_permission_preset(profile)
    resolution = resolve_profile(profile)

    assert expanded.model_dump() == profile.model_dump()
    assert resolution.effective_profile.prompt_profile == profile.prompt_profile
    assert wire_policy_bundle(profile).require_human_on_critical is True


def test_unknown_preset_fail_closed() -> None:
    with pytest.raises(ValidationError):
        GovernanceBundle(permission_preset="ultra_open")


def test_trusted_preset_cannot_liberalize_critical_hitl_authority() -> None:
    """Same upstream profile: TRUSTED must not widen HITL vs no preset."""
    shared_kwargs = {
        "approval_required": False,
        "autonomy": AutonomyLevel.AUTONOMOUS,
    }
    profile_no_preset = _profile_with_preset(None, **shared_kwargs)
    profile_trusted = _profile_with_preset(
        GovernancePermissionPreset.TRUSTED,
        **shared_kwargs,
    )

    bundle_no_preset = wire_policy_bundle(profile_no_preset)
    bundle_trusted = wire_policy_bundle(
        expand_governance_permission_preset(profile_trusted),
    )

    assert bundle_no_preset.require_human_on_critical is True
    assert bundle_trusted.require_human_on_critical is True


def test_effective_revision_digest_stability() -> None:
    profile = _profile_with_preset(
        GovernancePermissionPreset.RESTRICTED,
        tools=["read_file"],
    )
    first = resolve_profile(profile)
    second = resolve_profile(profile)
    assert first.fingerprint == second.fingerprint

    trusted = _profile_with_preset(GovernancePermissionPreset.TRUSTED, tools=["read_file"])
    trusted_resolution = resolve_profile(trusted)
    assert trusted_resolution.fingerprint != first.fingerprint
