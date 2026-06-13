# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox substrate resolution by isolation tier (ECC-4)."""

from __future__ import annotations

from intergrax.applications._shared.sandbox_host_wiring import resolve_hosted_sandbox_session
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.tools.providers.sandbox._session import resolve_sandbox_session
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_craft_sandbox_session(
    ctx: ToolWiringContext,
    profile: CodeCraftProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> SandboxExecCapable | None:
    """Route execution substrate per ``CodeCraftProfile.isolation_tier``."""
    if profile.isolation_tier == "cloud":
        integration_raw = ctx.extras.get("integration_profile")
        if isinstance(integration_raw, IntegrationProfile):
            hosted = resolve_hosted_sandbox_session(
                integration_raw,
                tenant_id=tenant_id,
                task_id=task_id,
            )
            if hosted is not None:
                return hosted
        if ctx.sandbox_host is not None:
            from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession

            return HostedSandboxSession.open(
                ctx.sandbox_host,
                tenant_id=tenant_id,
                task_id=task_id,
            )
    return resolve_sandbox_session(ctx)
