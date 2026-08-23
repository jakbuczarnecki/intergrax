# © Artur Czarnecki. All rights reserved.

"""PG-FIX-D — explicit typed policy action matching in RuntimePolicyBundleEvaluator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

_T0 = datetime(2026, 7, 21, 9, 0, 0, tzinfo=timezone.utc)
_ACTION = "CREATE_EXTERNAL_WORK"
_EVALUATOR_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "policy"
    / "runtime_policy_bundle_evaluator.py"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(action: str = _ACTION) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=action,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope-default",
        task_id="t1",
        run_id="r1",
        principal_id="u1",
    )


def _bundle(*rules: PolicyBundleRule):
    return build_immutable_runtime_policy_bundle(
        bundle_id="pg-fix-d-pack",
        version="1.0.0",
        rules=rules,
        issued_at=_T0,
    )


def test_d1_explicit_match_action_selects_rule() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="policy.create",
            effect="allow",
            match_action=_ACTION,
        )
    )
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    assert ev.decision.action is PolicyAction.ALLOW
    assert ev.matched_rule_id == "policy.create"


def test_d2_legacy_suffix_without_match_action_fails_closed() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="whatever.CREATE_EXTERNAL_WORK",
            effect="allow",
        )
    )
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    assert ev.decision.action is PolicyAction.DENY
    assert ev.matched_rule_id == "bundle.no_match"
    assert ev.decision.reason == "no_matching_rule"


def test_d3_rule_id_renaming_preserves_decision() -> None:
    rule_a = PolicyBundleRule(
        rule_id="policy.original.CREATE",
        effect="allow",
        match_action=_ACTION,
    )
    rule_b = PolicyBundleRule(
        rule_id="completely-unrelated-name",
        effect="allow",
        match_action=_ACTION,
    )
    ev_a = RuntimePolicyBundleEvaluator(
        _bundle(rule_a), clock=lambda: _T0
    ).evaluate(_request())
    ev_b = RuntimePolicyBundleEvaluator(
        _bundle(rule_b), clock=lambda: _T0
    ).evaluate(_request())
    assert ev_a.decision.action is PolicyAction.ALLOW
    assert ev_b.decision.action is PolicyAction.ALLOW
    assert ev_a.matched_rule_id == "policy.original.CREATE"
    assert ev_b.matched_rule_id == "completely-unrelated-name"


def test_d4_conflicting_rule_id_vs_match_action() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="looks-like.DELETE_EXTERNAL_WORK",
            effect="allow",
            match_action=_ACTION,
        )
    )
    evaluator = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    delete_ev = evaluator.evaluate(_request("DELETE_EXTERNAL_WORK"))
    create_ev = evaluator.evaluate(_request(_ACTION))
    assert delete_ev.decision.action is PolicyAction.DENY
    assert delete_ev.matched_rule_id == "bundle.no_match"
    assert create_ev.decision.action is PolicyAction.ALLOW
    assert create_ev.matched_rule_id == "looks-like.DELETE_EXTERNAL_WORK"


def test_d5_no_explicit_match_candidate_fails_closed() -> None:
    bundle = _bundle(
        PolicyBundleRule(rule_id="metadata-only", effect="allow"),
        PolicyBundleRule(rule_id="other", effect="deny"),
    )
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    assert ev.decision.action is PolicyAction.DENY
    assert ev.matched_rule_id == "bundle.no_match"


def test_d6_bundle_evidence_consistent_on_explicit_match() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="policy.create",
            effect="allow",
            match_action=_ACTION,
        )
    )
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    assert ev.bundle_id == bundle.bundle_id
    assert ev.bundle_version == bundle.version
    assert ev.bundle_digest == bundle.canonical_digest
    assert ev.matched_rule_id == "policy.create"
    assert ev.decision.policy_bundle_id == bundle.bundle_id
    assert ev.decision.policy_bundle_version == bundle.version
    assert ev.decision.policy_bundle_digest == bundle.canonical_digest
    assert ev.decision.policy_rule_id == "policy.create"
    assert ev.request_digest
    ev.assert_consistent_with_bundle(bundle)


def test_d7_explicit_match_action_changes_canonical_digest() -> None:
    suffix_only = _bundle(
        PolicyBundleRule(
            rule_id=f"legacy.{_ACTION}",
            effect="allow",
        )
    )
    explicit = _bundle(
        PolicyBundleRule(
            rule_id=f"legacy.{_ACTION}",
            effect="allow",
            match_action=_ACTION,
        )
    )
    assert suffix_only.canonical_digest != explicit.canonical_digest
    suffix_ev = RuntimePolicyBundleEvaluator(
        suffix_only, clock=lambda: _T0
    ).evaluate(_request())
    explicit_ev = RuntimePolicyBundleEvaluator(
        explicit, clock=lambda: _T0
    ).evaluate(_request())
    assert suffix_ev.decision.action is PolicyAction.DENY
    assert explicit_ev.decision.action is PolicyAction.ALLOW
    assert explicit_ev.bundle_digest == explicit.canonical_digest


def test_d9_evaluator_source_has_no_rule_id_applicability_parsing() -> None:
    source = _EVALUATOR_SOURCE.read_text(encoding="utf-8")
    assert "rule_id.endswith" not in source
    assert "rule_id.startswith" not in source
    assert "rule_id.split" not in source
    assert "re.search" not in source
    assert "re.match" not in source
