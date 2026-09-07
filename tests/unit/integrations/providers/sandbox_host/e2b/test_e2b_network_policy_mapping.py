# © Artur Czarnecki. All rights reserved.

"""E2B exact-host allowlist mapping unit tests."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxSecurityError
from intergrax.integrations.providers.sandbox_host.e2b.network_policy import (
    intergrax_hosts_to_e2b_allow_out,
    validate_security_allowlist_requirements,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = pytest.mark.unit


def test_exact_https_443_host_maps_to_provider_hostname() -> None:
    allowlist = canonicalize_network_egress_allowlist(["https://a.example.com:443"])
    payload = intergrax_hosts_to_e2b_allow_out(allowlist)
    assert payload.allow_out == ("a.example.com",)
    assert payload.deny_out == ("0.0.0.0/0",)


def test_multiple_hosts_map_to_deterministic_sorted_list() -> None:
    allowlist = canonicalize_network_egress_allowlist(
        ["https://b.example.com", "https://a.example.com"],
    )
    payload = intergrax_hosts_to_e2b_allow_out(allowlist)
    assert payload.allow_out == ("a.example.com", "b.example.com")


def test_duplicate_canonical_hosts_do_not_duplicate_provider_rules() -> None:
    allowlist = canonicalize_network_egress_allowlist(
        ["https://a.example.com", "https://a.example.com:443"],
    )
    payload = intergrax_hosts_to_e2b_allow_out(allowlist)
    assert payload.allow_out == ("a.example.com",)


def test_http_scheme_rejected() -> None:
    allowlist = canonicalize_network_egress_allowlist(["http://a.example.com"])
    with pytest.raises(E2bSandboxSecurityError, match="HTTPS only"):
        intergrax_hosts_to_e2b_allow_out(allowlist)


def test_non_443_port_rejected() -> None:
    allowlist = canonicalize_network_egress_allowlist(["https://a.example.com:8443"])
    with pytest.raises(E2bSandboxSecurityError, match="TLS:443"):
        intergrax_hosts_to_e2b_allow_out(allowlist)


def test_empty_allowlist_rejected() -> None:
    with pytest.raises(E2bSandboxSecurityError, match="must not be empty"):
        validate_security_allowlist_requirements(canonicalize_network_egress_allowlist([]))
