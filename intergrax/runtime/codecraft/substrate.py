# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed craft sandbox substrate resolution and capability evidence (AW-7B-GATE)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.codecraft.profile import CodeCraftProfile, IsolationTier, NetworkEgress
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.sandbox.contracts import SandboxExecCapable, SandboxSecurityCapable
from intergrax.runtime.sandbox.hosted_resolver import resolve_hosted_sandbox_session
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.sandbox._session import resolve_sandbox_session
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class CraftSubstrateCapabilities:
    """Runtime-proven substrate security capabilities — not profile text."""

    requested_tier: IsolationTier
    resolved_tier: IsolationTier
    provider_id: str
    downgraded: bool
    network_egress_enforced: bool


@dataclass(frozen=True, slots=True)
class CraftSandboxResolution:
    """Fail-closed craft sandbox resolution with typed substrate evidence."""

    session: SandboxExecCapable | None
    capabilities: CraftSubstrateCapabilities | None = None
    error: str = ""


def _egress_deny_proven(network_egress: NetworkEgress, session: SandboxExecCapable) -> bool:
    if network_egress != "deny":
        return True
    if not isinstance(session, SandboxSecurityCapable):
        return False
    evidence = session.security_capabilities()
    return evidence.network_egress_deny_enforced is True


def probe_substrate_capabilities(
    session: SandboxExecCapable,
    *,
    requested_tier: IsolationTier,
    network_egress: NetworkEgress,
) -> CraftSubstrateCapabilities:
    egress_enforced = _egress_deny_proven(network_egress, session)

    if isinstance(session, HostedSandboxSession):
        resolved_tier: IsolationTier = "cloud" if requested_tier == "cloud" else "container"
        provider_id = (
            session.security_capabilities().provider_id
            if isinstance(session, SandboxSecurityCapable)
            else f"hosted:{session.session_id}"
        )
        return CraftSubstrateCapabilities(
            requested_tier=requested_tier,
            resolved_tier=resolved_tier,
            provider_id=provider_id,
            downgraded=False,
            network_egress_enforced=egress_enforced,
        )

    if isinstance(session, SandboxSession):
        provider_id = (
            session.security_capabilities().provider_id
            if isinstance(session, SandboxSecurityCapable)
            else f"local:{session.session_id}"
        )
        return CraftSubstrateCapabilities(
            requested_tier=requested_tier,
            resolved_tier="local",
            provider_id=provider_id,
            downgraded=requested_tier in ("container", "cloud"),
            network_egress_enforced=egress_enforced,
        )

    return CraftSubstrateCapabilities(
        requested_tier=requested_tier,
        resolved_tier=requested_tier,
        provider_id="unknown",
        downgraded=False,
        network_egress_enforced=egress_enforced,
    )


def resolve_craft_sandbox(
    ctx: ToolWiringContext,
    profile: CodeCraftProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> CraftSandboxResolution:
    """Resolve execution substrate per isolation tier without silent downgrade."""
    requested = profile.isolation_tier

    if requested in ("cloud", "container"):
        integration_raw = ctx.extras.get("integration_profile")
        if isinstance(integration_raw, IntegrationProfile):
            hosted = resolve_hosted_sandbox_session(
                integration_raw,
                tenant_id=tenant_id,
                task_id=task_id,
            )
            if hosted is not None:
                capabilities = probe_substrate_capabilities(
                    hosted,
                    requested_tier=requested,
                    network_egress=profile.network_egress,
                )
                if profile.network_egress == "deny" and not capabilities.network_egress_enforced:
                    return CraftSandboxResolution(
                        session=None,
                        error="network_egress_requirement_unsatisfied",
                    )
                return CraftSandboxResolution(session=hosted, capabilities=capabilities)

        if ctx.sandbox_host is not None:
            hosted = HostedSandboxSession.open(
                ctx.sandbox_host,
                tenant_id=tenant_id,
                task_id=task_id,
            )
            capabilities = probe_substrate_capabilities(
                hosted,
                requested_tier=requested,
                network_egress=profile.network_egress,
            )
            if profile.network_egress == "deny" and not capabilities.network_egress_enforced:
                return CraftSandboxResolution(
                    session=None,
                    error="network_egress_requirement_unsatisfied",
                )
            return CraftSandboxResolution(session=hosted, capabilities=capabilities)

        return CraftSandboxResolution(
            session=None,
            error="isolation_requirement_unsatisfied",
        )

    local = resolve_sandbox_session(ctx)
    if local is None:
        return CraftSandboxResolution(session=None, error="sandbox_session_not_configured")

    capabilities = probe_substrate_capabilities(
        local,
        requested_tier="local",
        network_egress=profile.network_egress,
    )
    if profile.network_egress == "deny" and not capabilities.network_egress_enforced:
        return CraftSandboxResolution(
            session=None,
            error="network_egress_requirement_unsatisfied",
        )
    return CraftSandboxResolution(session=local, capabilities=capabilities)
