# © Artur Czarnecki. All rights reserved.

"""Typed sandbox network egress host-scope contract tests (AW-7C P0-1)."""

from __future__ import annotations

import pytest

from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressScopeError,
    allowlist_enforcement_satisfies_request,
    canonicalize_network_egress_allowlist,
    parse_network_egress_host,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APPROVED = parse_network_egress_host("https://api.example.com")
_APPROVED_ALT = parse_network_egress_host({"scheme": "https", "hostname": "data.example.com", "port": 443})


def test_canonicalize_deduplicates_and_orders() -> None:
    allowlist = canonicalize_network_egress_allowlist(
        [
            "https://b.example.com",
            "https://a.example.com",
            "https://a.example.com/",
        ],
    )
    assert [host.hostname for host in allowlist.hosts] == ["a.example.com", "b.example.com"]


def test_reject_empty_hostname() -> None:
    with pytest.raises(NetworkEgressScopeError):
        parse_network_egress_host({"scheme": "https", "hostname": ""})


def test_reject_wildcard_hostname() -> None:
    with pytest.raises(NetworkEgressScopeError):
        parse_network_egress_host("https://*.example.com")


def test_reject_localhost() -> None:
    with pytest.raises(NetworkEgressScopeError):
        parse_network_egress_host("https://localhost")


def test_reject_private_target() -> None:
    with pytest.raises(NetworkEgressScopeError):
        parse_network_egress_host("https://10.0.0.1")


def test_reject_embedded_credentials() -> None:
    with pytest.raises(NetworkEgressScopeError):
        parse_network_egress_host("https://user:pass@example.com")


def test_allowlist_enforcement_rejects_superset() -> None:
    requested = canonicalize_network_egress_allowlist(["https://a.example.com", "https://b.example.com"])
    enforced = canonicalize_network_egress_allowlist(
        ["https://a.example.com", "https://b.example.com", "https://c.example.com"],
    )
    assert allowlist_enforcement_satisfies_request(requested, enforced) is False


def test_allowlist_enforcement_accepts_exact_match() -> None:
    requested = canonicalize_network_egress_allowlist(["https://a.example.com"])
    enforced = canonicalize_network_egress_allowlist(["https://a.example.com"])
    assert allowlist_enforcement_satisfies_request(requested, enforced) is True


def test_allowlist_enforcement_accepts_restrictive_subset() -> None:
    requested = canonicalize_network_egress_allowlist(
        ["https://a.example.com", "https://b.example.com"],
    )
    enforced = canonicalize_network_egress_allowlist(["https://a.example.com"])
    assert allowlist_enforcement_satisfies_request(requested, enforced) is True


def test_fingerprint_is_stable() -> None:
    first = canonicalize_network_egress_allowlist(["https://a.example.com", "https://b.example.com"])
    second = canonicalize_network_egress_allowlist(["https://b.example.com", "https://a.example.com"])
    assert first.fingerprint() == second.fingerprint()
