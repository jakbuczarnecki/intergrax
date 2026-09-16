# © Artur Czarnecki. All rights reserved.

"""Apply runtime sandbox isolation authority to ``ToolWiringContext`` (composition-owned)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.contracts.execution_environment_isolation import (
    EffectiveProfileRevisionIsolationView,
    ProfileSandboxIsolationSource,
)
from intergrax.contracts.runtime_sandbox_isolation_authority import (
    RuntimeSandboxIsolationAuthority,
)
from intergrax.tools.registry.wiring import ToolWiringContext


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
