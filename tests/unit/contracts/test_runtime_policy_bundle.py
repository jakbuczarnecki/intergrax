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


def test_naive_issued_at_rejected() -> None:
    from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle

    naive = datetime(2026, 7, 20, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        ImmutableRuntimePolicyBundle(
            bundle_id="bundle-demo",
            version="1.0.0",
            rules=(PolicyBundleRule(rule_id="r1"),),
            issued_at=naive,
            canonical_digest="",
        ).compute_digest()


def test_with_canonical_digest_detects_mismatch() -> None:
    from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle

    bundle = ImmutableRuntimePolicyBundle(
        bundle_id="bundle-demo",
        version="1.0.0",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=_T0,
        canonical_digest="sha256:" + ("ff" * 32),
    )
    with pytest.raises(ValueError, match="canonical_digest does not match"):
        bundle.with_canonical_digest()


def test_bundle_id_and_issued_at_change_digest() -> None:
    a = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-a",
        version="1.0.0",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=_T0,
    )
    b = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-b",
        version="1.0.0",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=_T0,
    )
    c = build_immutable_runtime_policy_bundle(
        bundle_id="bundle-a",
        version="1.0.0",
        rules=(PolicyBundleRule(rule_id="r1"),),
        issued_at=datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert a.canonical_digest != b.canonical_digest
    assert a.canonical_digest != c.canonical_digest
