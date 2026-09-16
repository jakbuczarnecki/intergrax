# © Artur Czarnecki. All rights reserved.

"""Explicit runtime-host sandbox isolation authority (composition-owned, not session-derived)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intergrax.contracts.execution_environment_isolation import (
    EffectiveProfileRevisionIsolationView,
    ProfileSandboxIsolationSource,
)
from intergrax.contracts.sandbox_profile import SandboxProfile
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class RuntimeSandboxIsolationAuthority:
    """
    Narrow sandbox isolation authority for UAEP/Nexus runtime hosts without Tier-3 profile.

    Implements :class:`~intergrax.contracts.execution_environment_isolation.ProfileSandboxIsolationSource`.
    Must be supplied explicitly by composition — never inferred from ``SandboxExecCapable`` availability.
    """

    sandbox: SandboxProfile

    @classmethod
    def local_sandbox_exec_enabled(cls) -> RuntimeSandboxIsolationAuthority:
        """Explicit composition preset: local sandbox manager may run sandbox.exec."""
        return cls(sandbox=SandboxProfile(enable_exec_tool=True))


def wiring_has_sandbox_isolation_authority(ctx: ToolWiringContext) -> bool:
    """True when wiring already carries pinned, legacy profile, or explicit runtime authority."""
    if ctx.sandbox_isolation_authority is not None:
        return True
    revision_raw = ctx.extras.get("effective_profile_revision")
    if isinstance(revision_raw, EffectiveProfileRevisionIsolationView):
        return True
    legacy = ctx.extras.get("effective_environment_profile")
    return isinstance(legacy, ProfileSandboxIsolationSource)


def apply_runtime_sandbox_isolation_authority(
    ctx: ToolWiringContext,
    authority: RuntimeSandboxIsolationAuthority,
) -> ToolWiringContext:
    """Return wiring with explicit runtime sandbox authority (no override of existing)."""
    if wiring_has_sandbox_isolation_authority(ctx):
        return ctx
    return replace(ctx, sandbox_isolation_authority=authority)
