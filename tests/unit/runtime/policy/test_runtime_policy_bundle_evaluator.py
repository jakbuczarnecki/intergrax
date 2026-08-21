# © Artur Czarnecki. All rights reserved.

"""PC-1: bundle-backed policy evaluation."""

from __future__ import annotations

from datetime import datetime, timezone

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


def _bundle(**kwargs):
    rules = kwargs.pop(
        "rules",
        (
            PolicyBundleRule(
                rule_id="r.create",
                effect="allow",
                match_action="CREATE_EXTERNAL_WORK",
            ),
            PolicyBundleRule(
                rule_id="r.accept",
                effect="allow",
                match_action="ACCEPT_QUOTE",
            ),
        ),
    )
    return build_immutable_runtime_policy_bundle(
        bundle_id="eval-pack",
        version="1.0.0",
        rules=rules,
        issued_at=_T0,
        **kwargs,
    )


def _request(action: str = "CREATE_EXTERNAL_WORK") -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=action,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope-default",
        task_id="t1",
        run_id="r1",
        principal_id="u1",
    )


def test_decision_bound_to_evaluated_bundle() -> None:
    bundle = _bundle()
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    assert ev.bundle_digest == bundle.canonical_digest
    assert ev.decision.policy_bundle_digest == bundle.canonical_digest
    assert ev.matched_rule_id == "r.create"
    assert ev.decision.action is PolicyAction.ALLOW
    ev.assert_consistent_with_bundle(bundle)


def test_no_matching_rule_denies() -> None:
    bundle = _bundle()
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(
        _request("UNKNOWN_ACTION")
    )
    assert ev.decision.action is PolicyAction.DENY
    assert ev.matched_rule_id == "bundle.no_match"


def test_action_inconsistent_with_rule_fails_assert() -> None:
    from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
    from intergrax.contracts.runtime_policy import PolicyDecision

    bundle = _bundle(
        rules=(
            PolicyBundleRule(
                rule_id="r.deny",
                effect="deny",
                match_action="CREATE_EXTERNAL_WORK",
            ),
        )
    )
    # Construct an EvaluatedPolicyDecision that claims ALLOW for a deny rule
    # with matching pack identity (digest of this deny pack).
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        policy_rule_id="r.deny",
        policy_bundle_id=bundle.bundle_id,
        policy_bundle_version=bundle.version,
        policy_bundle_digest=bundle.canonical_digest,
        decision_id="tampered",
    )
    bad = EvaluatedPolicyDecision(
        decision=decision,
        bundle_id=bundle.bundle_id,
        bundle_version=bundle.version,
        bundle_digest=bundle.canonical_digest,
        matched_rule_id="r.deny",
        evaluated_at=_T0,
        request_digest="sha256:" + ("11" * 32),
    )
    with pytest.raises(ValueError, match="decision_action_mismatch_with_rule"):
        bad.assert_consistent_with_bundle(bundle)


def test_cannot_stamp_bundle_identity_after_evaluation() -> None:
    bundle = _bundle()
    ev = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0).evaluate(_request())
    stamped = ev.decision.model_copy(
        update={"policy_bundle_digest": "sha256:" + ("00" * 32)}
    )
    from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision

    with pytest.raises(ValueError, match="decision_bundle_digest_mismatch"):
        EvaluatedPolicyDecision(
            decision=stamped,
            bundle_id=ev.bundle_id,
            bundle_version=ev.bundle_version,
            bundle_digest=ev.bundle_digest,
            matched_rule_id=ev.matched_rule_id,
            evaluated_at=ev.evaluated_at,
            request_digest=ev.request_digest,
        )


def test_request_digest_stable_for_same_request() -> None:
    bundle = _bundle()
    evaluator = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    a = evaluator.evaluate(_request())
    b = evaluator.evaluate(_request())
    assert a.request_digest == b.request_digest
