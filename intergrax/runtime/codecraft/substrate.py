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
from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressAllowlist,
    allowlist_enforcement_satisfies_request,
)
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
    network_egress_deny_enforced: bool
    network_egress_allowlist_enforced: bool
    enforced_network_hosts: NetworkEgressAllowlist | None = None

    @property
    def network_egress_enforced(self) -> bool:
        """Backward-compatible aggregate for the active requested egress mode."""
        if self.network_egress_allowlist_enforced:
            return True
        return self.network_egress_deny_enforced


@dataclass(frozen=True, slots=True)
class CraftSandboxResolution:
    """Fail-closed craft sandbox resolution with typed substrate evidence."""

    session: SandboxExecCapable | None
    capabilities: CraftSubstrateCapabilities | None = None
    error: str = ""


def _security_evidence(session: SandboxExecCapable):
    if not isinstance(session, SandboxSecurityCapable):
        return None
    return session.security_capabilities()


def _probe_network_egress_capabilities(
    network_egress: NetworkEgress,
    requested_allowlist: NetworkEgressAllowlist,
    session: SandboxExecCapable,
) -> tuple[bool, bool, bool, NetworkEgressAllowlist | None]:
    evidence = _security_evidence(session)
    if evidence is None:
        return False, False, False, None

    deny_enforced = evidence.network_egress_deny_enforced is True
    allowlist_enforced = evidence.network_egress_allowlist_enforced is True
    enforced_hosts = evidence.enforced_network_hosts

    if network_egress == "deny":
        return deny_enforced, deny_enforced, False, None

    if network_egress == "allowlist":
        if not allowlist_enforced or enforced_hosts is None:
            return False, False, False, enforced_hosts
        if not allowlist_enforcement_satisfies_request(requested_allowlist, enforced_hosts):
            return False, False, False, enforced_hosts
        return True, False, True, enforced_hosts

    return False, False, False, None


def _network_egress_resolution_error(network_egress: NetworkEgress) -> str:
    if network_egress == "allowlist":
        return "network_egress_allowlist_requirement_unsatisfied"
    return "network_egress_requirement_unsatisfied"


def probe_substrate_capabilities(
    session: SandboxExecCapable,
    *,
    requested_tier: IsolationTier,
    network_egress: NetworkEgress,
    requested_allowlist: NetworkEgressAllowlist | None = None,
) -> CraftSubstrateCapabilities:
    allowlist = requested_allowlist or NetworkEgressAllowlist(hosts=())
    requirement_satisfied, deny_enforced, allowlist_enforced, enforced_hosts = _probe_network_egress_capabilities(
        network_egress,
        allowlist,
        session,
    )
    _ = requirement_satisfied

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
            network_egress_deny_enforced=deny_enforced,
            network_egress_allowlist_enforced=allowlist_enforced,
            enforced_network_hosts=enforced_hosts,
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
            network_egress_deny_enforced=deny_enforced,
            network_egress_allowlist_enforced=allowlist_enforced,
            enforced_network_hosts=enforced_hosts,
        )

    return CraftSubstrateCapabilities(
        requested_tier=requested_tier,
        resolved_tier=requested_tier,
        provider_id="unknown",
        downgraded=False,
        network_egress_deny_enforced=deny_enforced,
        network_egress_allowlist_enforced=allowlist_enforced,
        enforced_network_hosts=enforced_hosts,
    )


def _network_egress_requirement_satisfied(
    profile: CodeCraftProfile,
    capabilities: CraftSubstrateCapabilities,
) -> bool:
    if profile.network_egress == "deny":
        return capabilities.network_egress_deny_enforced
    if profile.network_egress == "allowlist":
        return capabilities.network_egress_allowlist_enforced
    return False


def resolve_craft_sandbox(
    ctx: ToolWiringContext,
    profile: CodeCraftProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> CraftSandboxResolution:
    """Resolve execution substrate per isolation tier without silent downgrade."""
    requested = profile.isolation_tier
    requested_allowlist = profile.network_egress_allowlist_scope

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
                    requested_allowlist=requested_allowlist,
                )
                if not _network_egress_requirement_satisfied(profile, capabilities):
                    return CraftSandboxResolution(
                        session=None,
                        error=_network_egress_resolution_error(profile.network_egress),
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
                requested_allowlist=requested_allowlist,
            )
            if not _network_egress_requirement_satisfied(profile, capabilities):
                return CraftSandboxResolution(
                    session=None,
                    error=_network_egress_resolution_error(profile.network_egress),
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
        requested_allowlist=requested_allowlist,
    )
    if not _network_egress_requirement_satisfied(profile, capabilities):
        return CraftSandboxResolution(
            session=None,
            error=_network_egress_resolution_error(profile.network_egress),
        )
    return CraftSandboxResolution(session=local, capabilities=capabilities)
