# © Artur Czarnecki. All rights reserved.

"""E2B provider-state attestation unit tests."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxSecurityError
from intergrax.integrations.providers.sandbox_host.e2b.network_policy import (
    E2bProviderNetworkState,
    provider_state_to_enforced_allowlist,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = pytest.mark.unit

_REQUESTED = canonicalize_network_egress_allowlist(["https://approved.example.com"])


def _state(*allow: str, deny: tuple[str, ...] = ("0.0.0.0/0",)) -> E2bProviderNetworkState:
    return E2bProviderNetworkState(allow_out=tuple(allow), deny_out=deny)


def test_provider_exact_host_set_passes() -> None:
    enforced = provider_state_to_enforced_allowlist(
        _state("approved.example.com"),
        requested=_REQUESTED,
    )
    assert enforced.hosts[0].hostname == "approved.example.com"
    assert enforced.hosts[0].port == 443


def test_provider_extra_host_fails() -> None:
    with pytest.raises(E2bSandboxSecurityError):
        provider_state_to_enforced_allowlist(
            _state("approved.example.com", "extra.example.com"),
            requested=_REQUESTED,
        )


def test_provider_wildcard_fails() -> None:
    with pytest.raises(E2bSandboxSecurityError, match="wildcard"):
        provider_state_to_enforced_allowlist(
            _state("*.example.com"),
            requested=_REQUESTED,
        )


def test_provider_cidr_fails() -> None:
    with pytest.raises(E2bSandboxSecurityError, match="CIDR"):
        provider_state_to_enforced_allowlist(
            _state("10.0.0.0/8"),
            requested=_REQUESTED,
        )


def test_provider_missing_network_state_fails() -> None:
    with pytest.raises(E2bSandboxSecurityError, match="missing allowOut"):
        provider_state_to_enforced_allowlist(_state(), requested=_REQUESTED)


def test_provider_missing_default_deny_fails() -> None:
    with pytest.raises(E2bSandboxSecurityError, match="default-deny denyOut"):
        provider_state_to_enforced_allowlist(
            _state("approved.example.com", deny=()),
            requested=_REQUESTED,
        )


def test_request_echo_is_not_used_as_evidence() -> None:
    requested = canonicalize_network_egress_allowlist(["https://requested.example.com"])
    with pytest.raises(E2bSandboxSecurityError):
        provider_state_to_enforced_allowlist(
            _state("different.example.com"),
            requested=requested,
        )
