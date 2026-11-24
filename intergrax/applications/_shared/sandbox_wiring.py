# © Artur Czarnecki. All rights reserved.

"""Sandbox session manager wiring from environment profile (Phase H-APP.3.5)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.sandbox_profile import SandboxProfile
from intergrax.runtime.sandbox.manager import ENV_SANDBOX_ROOT, SandboxSessionManager
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry.catalog import UnknownToolBundleError, get_bundle, list_catalog_tool_ids

_SIDE_EFFECT_TOOL_PREFIXES = ("sandbox.", "workspace.", "gitlab.", "websearch.")


class SandboxCompositionError(ValueError):
    """Raised when guarded composition cannot bind sandbox isolation."""


def _is_side_effect_tool_id(tool_id: str) -> bool:
    return tool_id.startswith(_SIDE_EFFECT_TOOL_PREFIXES) or tool_id == "sandbox.exec"


def _effective_enabled_tool_ids(
    profile: ToolProfile,
    *,
    strict_bundle_resolution: bool,
) -> set[str]:
    enabled = set(profile.enabled)
    if profile.register_all_catalog_bundles:
        enabled.update(list_catalog_tool_ids())
        return enabled
    for bundle_id in profile.enabled_bundles:
        try:
            entry = get_bundle(bundle_id)
        except UnknownToolBundleError:
            if strict_bundle_resolution:
                raise
            continue
        enabled.update(entry.tool_ids)
    return enabled


def _profile_has_side_effect_tools(
    profile: ToolProfile,
    *,
    strict_bundle_resolution: bool,
) -> bool:
    for tool_id in _effective_enabled_tool_ids(
        profile,
        strict_bundle_resolution=strict_bundle_resolution,
    ):
        if _is_side_effect_tool_id(tool_id):
            return True
    return False


def sandbox_required_for_posture(env: ApplicationEnvironmentProfile) -> bool:
    """Guarded postures with effective side-effect tool enablement require explicit sandbox."""
    if not wiring_fails_closed(env):
        return False
    return _profile_has_side_effect_tools(env.tool_profile, strict_bundle_resolution=True)


def product_requires_sandbox(env: ApplicationEnvironmentProfile) -> bool:
    """AUDIT-IDEAL-11.1 — product hosts with side-effect tools require sandbox wiring."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return False
    return _profile_has_side_effect_tools(
        env.tool_profile,
        strict_bundle_resolution=True,
    )


def _resolve_guarded_sandbox_root(sandbox_profile: SandboxProfile) -> Path:
    if sandbox_profile.root is not None:
        return sandbox_profile.root
    env_root = os.environ.get(ENV_SANDBOX_ROOT, "").strip()
    if env_root:
        return Path(env_root)
    raise SandboxCompositionError(
        "guarded sandbox composition requires explicit sandbox root (SandboxProfile.root or INTERGRAX_SANDBOX_ROOT)",
    )


def wire_sandbox_sessions(
    env: ApplicationEnvironmentProfile,
) -> SandboxSessionManager | None:
    """Bind sandbox manager; enable ``sandbox.exec`` on tool profile when configured."""
    sandbox_profile = env.sandbox
    requires_isolation = sandbox_required_for_posture(env)
    guarded_exec_tool = wiring_fails_closed(env) and sandbox_profile is not None and sandbox_profile.enable_exec_tool
    if requires_isolation:
        if sandbox_profile is None:
            raise SandboxCompositionError(
                "guarded composition requires SandboxProfile when side-effect tools are enabled",
            )
        root = _resolve_guarded_sandbox_root(sandbox_profile)
        return SandboxSessionManager(root=root)
    if guarded_exec_tool and sandbox_profile is not None:
        root = _resolve_guarded_sandbox_root(sandbox_profile)
        return SandboxSessionManager(root=root)
    if sandbox_profile is None:
        return None
    return SandboxSessionManager(root=sandbox_profile.root)


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
