# © Artur Czarnecki. All rights reserved.

"""ImmutableRuntimePolicyBundle digest determinism."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)

pytestmark = [pytest.mark.unit]

_T0 = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def test_bundle_digest_stable_and_order_sensitive() -> None:
    rules_a = (
        PolicyBundleRule(rule_id="r1", description="allow create", effect="allow"),
        PolicyBundleRule(rule_id="r2", description="allow accept", effect="allow"),
    )
    rules_b = (
        PolicyBundleRule(rule_id="r2", description="allow accept", effect="allow"),
        PolicyBundleRule(rule_id="r1", description="allow create", effect="allow"),
    )
    a = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-demo",
        version="1.0.0",
        rules=rules_a,
        issued_at=_T0,
    )
    a2 = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-demo",
        version="1.0.0",
        rules=rules_a,
        issued_at=_T0,
    )
    b = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-demo",
        version="1.0.0",
        rules=rules_b,
        issued_at=_T0,
    )
    assert a.canonical_digest.startswith("sha256:")
    assert a.canonical_digest == a2.canonical_digest
    assert a.canonical_digest != b.canonical_digest


def test_bundle_field_change_changes_digest() -> None:
    base = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-demo",
        version="1.0.0",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=_T0,
    )
    other = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-demo",
        version="1.0.1",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=_T0,
    )
    assert base.canonical_digest != other.canonical_digest
