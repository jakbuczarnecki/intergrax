# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 marketplace handoff reference contract tests."""

from __future__ import annotations

import hashlib

import pytest

from intergrax.contracts.tools.marketplace_handoff_reference import (
    MARKETPLACE_GAP_HANDOFF_V2_PREFIX,
    MarketplaceHandoffReferenceError,
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
    parse_marketplace_domain_handoff_reference,
)

pytestmark = pytest.mark.unit

_KNOWN_DIGEST = "34d6107ee3909fc264136d0505981e1ddbe8c1a845617d49e448830d018fe170"


def test_same_tenant_operation_same_id() -> None:
    a = derive_marketplace_gap_tool_handoff_id(tenant_id="t1", operation_id="op")
    b = derive_marketplace_gap_tool_handoff_id(tenant_id="t1", operation_id="op")
    assert a == b


def test_different_tenant_same_operation_different_id() -> None:
    a = derive_marketplace_gap_tool_handoff_id(tenant_id="t1", operation_id="op")
    b = derive_marketplace_gap_tool_handoff_id(tenant_id="t2", operation_id="op")
    assert a != b


def test_different_operation_same_tenant_different_id() -> None:
    a = derive_marketplace_gap_tool_handoff_id(tenant_id="t1", operation_id="op1")
    b = derive_marketplace_gap_tool_handoff_id(tenant_id="t1", operation_id="op2")
    assert a != b


def test_deterministic_repeated_calls() -> None:
    values = [
        derive_marketplace_gap_tool_handoff_id(tenant_id="tenant", operation_id="x")
        for _ in range(5)
    ]
    assert len(set(values)) == 1


def test_v2_prefix() -> None:
    handoff_id = derive_marketplace_gap_tool_handoff_id(tenant_id="t", operation_id="o")
    assert handoff_id.startswith(MARKETPLACE_GAP_HANDOFF_V2_PREFIX)


def test_known_digest_vector() -> None:
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-a",
        operation_id="op-1",
    )
    assert handoff_id == f"{MARKETPLACE_GAP_HANDOFF_V2_PREFIX}{_KNOWN_DIGEST}"
    payload = (
        "intergrax.marketplace-gap-handoff.v2"
        + "\0"
        + "tenant-a"
        + "\0"
        + "op-1"
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == _KNOWN_DIGEST


def test_no_raw_tenant_substring() -> None:
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="super-secret-tenant",
        operation_id="op",
    )
    assert "super-secret-tenant" not in handoff_id


def test_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError):
        derive_marketplace_gap_tool_handoff_id(tenant_id="", operation_id="op")


def test_empty_operation_rejected() -> None:
    with pytest.raises(ValueError):
        derive_marketplace_gap_tool_handoff_id(tenant_id="t", operation_id="")


def test_domain_handoff_reference_round_trip() -> None:
    handoff_id = derive_marketplace_gap_tool_handoff_id(tenant_id="t", operation_id="o")
    ref = marketplace_domain_handoff_reference(handoff_id)
    assert parse_marketplace_domain_handoff_reference(ref) == handoff_id


def test_malformed_scheme_rejected() -> None:
    with pytest.raises(MarketplaceHandoffReferenceError):
        parse_marketplace_domain_handoff_reference("artifact://x")


def test_empty_handoff_in_reference_rejected() -> None:
    with pytest.raises(MarketplaceHandoffReferenceError):
        parse_marketplace_domain_handoff_reference("handoff://")
