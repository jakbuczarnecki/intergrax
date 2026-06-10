# © Artur Czarnecki. All rights reserved.

"""Sandbox session manager wiring from environment profile (Phase H-APP.3.5)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.tools.registry.profile import ToolProfile


def wire_sandbox_sessions(
    env: ApplicationEnvironmentProfile,
) -> SandboxSessionManager | None:
    """Bind sandbox manager; enable ``sandbox.exec`` on tool profile when configured."""
    sandbox_profile = env.sandbox
    if sandbox_profile is None:
        return None
    return SandboxSessionManager(root=sandbox_profile.root)


_SIDE_EFFECT_TOOL_PREFIXES = ("sandbox.", "workspace.", "gitlab.", "websearch.")


def product_requires_sandbox(env: ApplicationEnvironmentProfile) -> bool:
    """AUDIT-IDEAL-11.1 — product hosts with side-effect tools require sandbox wiring."""
    from intergrax.applications.contracts.application_host import ApplicationProfile

    if env.application_profile is not ApplicationProfile.PRODUCT:
        return False
    enabled = set(env.tool_profile.enabled)
    for bundle in env.tool_profile.enabled_bundles:
        enabled.add(bundle)
    return any(
        tool_id.startswith(_SIDE_EFFECT_TOOL_PREFIXES) or tool_id == "sandbox.exec"
        for tool_id in enabled
    )


def tool_profile_with_sandbox(env: ApplicationEnvironmentProfile) -> ToolProfile:
    """Return tool profile including ``sandbox.exec`` when sandbox is enabled."""
    profile = env.tool_profile
    sandbox_profile = env.sandbox
    if sandbox_profile is None or not sandbox_profile.enable_exec_tool:
        return profile
    if "sandbox.exec" in profile.enabled:
        return profile
    enabled = list(profile.enabled) + ["sandbox.exec"]
    return profile.model_copy(update={"enabled": enabled})
