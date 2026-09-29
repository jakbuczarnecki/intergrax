# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox attestation validation for scoped adaptive integration (AW-7C-P4)."""

from __future__ import annotations

from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities


class ScopedAdaptiveIntegrationSandboxSecurityError(Exception):
    """Fail-closed sandbox security attestation rejection."""


def validate_qualified_allowlist_attestation(
    *,
    qualified_allowlist: NetworkEgressAllowlist,
    capabilities: SandboxSecurityCapabilities,
) -> None:
    if capabilities.network_egress_allowlist_enforced is not True:
        raise ScopedAdaptiveIntegrationSandboxSecurityError(
            "network_egress_allowlist_enforced is not True",
        )
    enforced = capabilities.enforced_network_hosts
    if enforced is None:
        raise ScopedAdaptiveIntegrationSandboxSecurityError(
            "enforced_network_hosts missing",
        )
    qualified_hosts = {host.canonical_form() for host in qualified_allowlist.hosts}
    enforced_hosts = {host.canonical_form() for host in enforced.hosts}
    if not qualified_hosts and not enforced_hosts:
        return
    if not qualified_hosts:
        if enforced_hosts:
            raise ScopedAdaptiveIntegrationSandboxSecurityError(
                "deny-only scope cannot admit enforced hosts",
            )
        return
    missing = qualified_hosts - enforced_hosts
    if missing:
        raise ScopedAdaptiveIntegrationSandboxSecurityError(
            f"required hosts not enforced: {sorted(missing)}",
        )
    extra = enforced_hosts - qualified_hosts
    if extra:
        raise ScopedAdaptiveIntegrationSandboxSecurityError(
            f"enforced hosts broader than qualified: {sorted(extra)}",
        )


__all__ = [
    "ScopedAdaptiveIntegrationSandboxSecurityError",
    "validate_qualified_allowlist_attestation",
]
