# © Artur Czarnecki. All rights reserved.

"""AW-7C-P4 sandbox attestation validation."""

from __future__ import annotations

import pytest

from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.scoped_adaptive_integration_sandbox_validation import (
    ScopedAdaptiveIntegrationSandboxSecurityError,
    validate_qualified_allowlist_attestation,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities

pytestmark = pytest.mark.unit

_HOST_A = NetworkEgressHost(scheme="https", hostname="a.example.com", port=443)
_HOST_B = NetworkEgressHost(scheme="https", hostname="b.example.com", port=443)
_QUALIFIED = NetworkEgressAllowlist(hosts=(_HOST_A,))


def _capabilities(
    *,
    enforced: NetworkEgressAllowlist | None,
    allowlist_enforced: bool | None = True,
) -> SandboxSecurityCapabilities:
    return SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="test",
        network_egress_allowlist_enforced=allowlist_enforced,
        enforced_network_hosts=enforced,
    )


def test_exact_allowlist_attested_proceeds() -> None:
    validate_qualified_allowlist_attestation(
        qualified_allowlist=_QUALIFIED,
        capabilities=_capabilities(enforced=_QUALIFIED),
    )


def test_broader_attested_allowlist_rejected() -> None:
    with pytest.raises(ScopedAdaptiveIntegrationSandboxSecurityError):
        validate_qualified_allowlist_attestation(
            qualified_allowlist=_QUALIFIED,
            capabilities=_capabilities(
                enforced=NetworkEgressAllowlist(hosts=(_HOST_A, _HOST_B)),
            ),
        )


def test_allowlist_enforcement_false_rejected() -> None:
    with pytest.raises(ScopedAdaptiveIntegrationSandboxSecurityError):
        validate_qualified_allowlist_attestation(
            qualified_allowlist=_QUALIFIED,
            capabilities=_capabilities(enforced=_QUALIFIED, allowlist_enforced=False),
        )


def test_enforced_hosts_none_rejected() -> None:
    with pytest.raises(ScopedAdaptiveIntegrationSandboxSecurityError):
        validate_qualified_allowlist_attestation(
            qualified_allowlist=_QUALIFIED,
            capabilities=_capabilities(enforced=None),
        )
