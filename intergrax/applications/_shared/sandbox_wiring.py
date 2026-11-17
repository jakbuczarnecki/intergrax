# © Artur Czarnecki. All rights reserved.

"""Sandbox session manager wiring from environment profile (Phase H-APP.3.5)."""

from __future__ import annotations

import os

from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.sandbox.manager import ENV_SANDBOX_ROOT, SandboxSessionManager
from intergrax.tools.registry.profile import ToolProfile

_SIDE_EFFECT_TOOL_PREFIXES = ("sandbox.", "workspace.", "gitlab.", "websearch.")


class SandboxCompositionError(RuntimeError):
    """Raised when a fail-closed posture enables side-effect tools without an explicit sandbox root."""


def _has_side_effect_tools(env: ApplicationEnvironmentProfile) -> bool:
    enabled = set(env.tool_profile.enabled)
    for bundle in env.tool_profile.enabled_bundles:
        enabled.add(bundle)
    return any(tool_id.startswith(_SIDE_EFFECT_TOOL_PREFIXES) or tool_id == "sandbox.exec" for tool_id in enabled)


def sandbox_required_for_posture(env: ApplicationEnvironmentProfile) -> bool:
    """PRODUCT or STRICT posture (DP-24) with side-effect tools enabled requires a sandbox."""
    return wiring_fails_closed(env) and _has_side_effect_tools(env)


def wire_sandbox_sessions(
    env: ApplicationEnvironmentProfile,
) -> SandboxSessionManager | None:
    """Bind sandbox manager; enable ``sandbox.exec`` on tool profile when configured.

    In a fail-closed posture with side-effect tools, a configured sandbox profile with an
    explicit root (profile root or ``INTERGRAX_SANDBOX_ROOT``) is required; the relative
    default root is never used silently.
    """
    sandbox_profile = env.sandbox
    required = sandbox_required_for_posture(env)
    if sandbox_profile is None:
        if required:
            raise SandboxCompositionError(
                "side-effect tools are enabled in a PRODUCT/STRICT posture but no sandbox profile is configured"
            )
        return None
    if required and sandbox_profile.root is None and not os.environ.get(ENV_SANDBOX_ROOT, "").strip():
        raise SandboxCompositionError(
            "sandbox composition in a PRODUCT/STRICT posture requires an explicit root "
            f"(SandboxProfile.root or {ENV_SANDBOX_ROOT})"
        )
    return SandboxSessionManager(root=sandbox_profile.root)


def product_requires_sandbox(env: ApplicationEnvironmentProfile) -> bool:
    """AUDIT-IDEAL-11.1 — product hosts with side-effect tools require sandbox wiring."""
    from intergrax.applications.contracts.application_host import ApplicationProfile

    if env.application_profile is not ApplicationProfile.PRODUCT:
        return False
    return _has_side_effect_tools(env)


def tool_profile_with_sandbox(env: ApplicationEnvironmentProfile) -> ToolProfile:
    """Return tool profile including ``sandbox.exec`` when sandbox is enabled."""
    from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID, SANDBOX_EXEC_TOOL_ID

    profile = env.tool_profile
    sandbox_profile = env.sandbox
    if sandbox_profile is None or not sandbox_profile.enable_exec_tool:
        return profile
    if SANDBOX_EXEC_TOOL_ID in profile.enabled:
        return profile
    if SANDBOX_BUNDLE_ID in profile.enabled_bundles:
        return profile
    if profile.enabled_bundles and not profile.enabled:
        bundles = list(profile.enabled_bundles)
        if SANDBOX_BUNDLE_ID not in bundles:
            bundles.append(SANDBOX_BUNDLE_ID)
        return profile.model_copy(update={"enabled_bundles": bundles})
    enabled = list(profile.enabled) + [SANDBOX_EXEC_TOOL_ID]
    return profile.model_copy(update={"enabled": enabled})
