# © Artur Czarnecki. All rights reserved.

"""CodeCraftProfile network egress scope validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.sandbox.network_egress import parse_network_egress_host

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HOST = parse_network_egress_host("https://api.example.com")


def test_deny_with_empty_scope_valid() -> None:
    profile = CodeCraftProfile(mode="autonomous", network_egress="deny")
    assert profile.network_egress_allowlist == ()


def test_deny_with_allowlist_scope_invalid() -> None:
    with pytest.raises(ValidationError, match="network_egress=deny cannot include"):
        CodeCraftProfile(
            mode="autonomous",
            network_egress="deny",
            network_egress_allowlist=(_HOST,),
        )


def test_allowlist_with_empty_scope_invalid() -> None:
    with pytest.raises(ValidationError, match="network_egress=allowlist requires"):
        CodeCraftProfile(mode="autonomous", network_egress="allowlist")


def test_allowlist_with_valid_hosts_valid() -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        network_egress="allowlist",
        network_egress_allowlist=(_HOST,),
    )
    assert profile.network_egress_allowlist_scope.fingerprint().startswith("sha256:")


def test_invalid_hostname_rejected() -> None:
    with pytest.raises(ValidationError):
        CodeCraftProfile(
            mode="autonomous",
            network_egress="allowlist",
            network_egress_allowlist=("https://localhost",),
        )
