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
