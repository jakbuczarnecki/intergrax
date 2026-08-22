# © Artur Czarnecki. All rights reserved.

"""GEC-5 — platform meaningful side-effect policy evaluation."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


def _request(**overrides: object) -> MeaningfulSideEffectRequest:
    payload: dict[str, object] = {
        "action": "ACCEPT_QUOTE",
        "kinds": (MeaningfulSideEffectKind.COMMITMENT, MeaningfulSideEffectKind.MUTATION),
        "side_effect_scope_id": "scope-gec5",
        "task_id": "task-1",
        "run_id": "run-1",
        "principal_id": "user-1",
        "tenant_id": "tenant-a",
        "external_target": "provider-x",
    }
    payload.update(overrides)
    return MeaningfulSideEffectRequest.model_validate(payload)


def _rule(**overrides: object) -> MeaningfulSideEffectPolicyRule:
    payload: dict[str, object] = {
        "rule_id": "gec5.rule",
        "decision": PolicyAction.ALLOW,
        "action": "ACCEPT_QUOTE",
    }
    payload.update(overrides)
    return MeaningfulSideEffectPolicyRule(**payload)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.gate
def test_blank_rule_id_rejected() -> None:
    with pytest.raises(ValueError, match="rule_id must be non-empty"):
        MeaningfulSideEffectPolicyRule(
            rule_id="   ",
            decision=PolicyAction.ALLOW,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_blank_action_filter_rejected() -> None:
    with pytest.raises(ValueError, match="action must be non-empty"):
        MeaningfulSideEffectPolicyRule(
            rule_id="gec5.blank_action",
            decision=PolicyAction.ALLOW,
            action="   ",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_rule_is_immutable() -> None:
    rule = _rule()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.rule_id = "other"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_invalid_decision_string_not_representable() -> None:
    with pytest.raises(TypeError):
        MeaningfulSideEffectPolicyRule(
            rule_id="gec5.bad",
            decision="foobar",  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.gate
def test_default_fail_closed_without_matching_rule() -> None:
    engine = RuntimePolicyEngine()
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_indeterminate"
    assert decision.policy_rule_id == "default.meaningful_side_effect.indeterminate"


@pytest.mark.unit
@pytest.mark.gate
def test_missing_identity_fail_closed() -> None:
    engine = RuntimePolicyEngine(meaningful_side_effect_rules=(_rule(),))
    request = MeaningfulSideEffectRequest.model_construct(
        action="ACCEPT_QUOTE",
        kinds=(MeaningfulSideEffectKind.COMMITMENT,),
        task_id="",
        run_id="run-1",
        principal_id="user-1",
    )
    decision = engine.evaluate_meaningful_side_effect(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_identity_missing"
    assert decision.policy_rule_id == "default.meaningful_side_effect.identity"


@pytest.mark.unit
@pytest.mark.gate
def test_missing_principal_fail_closed() -> None:
    engine = RuntimePolicyEngine(meaningful_side_effect_rules=(_rule(),))
    decision = engine.evaluate_meaningful_side_effect(_request(principal_id=None))
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_principal_missing"
    assert decision.policy_rule_id == "default.meaningful_side_effect.principal"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_allow_deny_require_human() -> None:
    allow_engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.allow",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.ALLOW,
            ),
        )
    )
    allow_decision = allow_engine.evaluate_meaningful_side_effect(_request())
    assert allow_decision.action is PolicyAction.ALLOW
    assert allow_decision.policy_rule_id == "gec5.allow"
    assert allow_decision.reason == "meaningful_side_effect:allow"

    deny_engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.deny",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.DENY,
            ),
        )
    )
    assert (
        deny_engine.evaluate_meaningful_side_effect(_request()).action
        is PolicyAction.DENY
    )

    human_engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.hitl",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.REQUIRE_HUMAN,
            ),
        )
    )
    assert (
        human_engine.evaluate_meaningful_side_effect(_request()).action
        is PolicyAction.REQUIRE_HUMAN
    )


@pytest.mark.unit
@pytest.mark.gate
def test_escalate_preserved() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.escalate",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.ESCALATE,
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.ESCALATE
    assert decision.policy_rule_id == "gec5.escalate"


@pytest.mark.unit
@pytest.mark.gate
def test_modify_unsupported_fail_closed() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.modify",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.MODIFY,
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_unsupported_decision"
    assert decision.policy_rule_id == "gec5.modify"


@pytest.mark.unit
@pytest.mark.gate
def test_action_filtering() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.other_action",
                action="CREATE_EXTERNAL_WORK",
                decision=PolicyAction.ALLOW,
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_indeterminate"


@pytest.mark.unit
@pytest.mark.gate
def test_same_specificity_deny_dominates_allow() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.first",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.DENY,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.second",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.ALLOW,
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "gec5.first"


@pytest.mark.unit
@pytest.mark.gate
def test_unrestricted_action_rule_matches_any_action() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.any",
                decision=PolicyAction.ALLOW,
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request(action="OTHER_ACTION"))
    assert decision.action is PolicyAction.ALLOW
    assert decision.policy_rule_id == "gec5.any"


@pytest.mark.unit
@pytest.mark.gate
def test_explicit_reason_preserved() -> None:
    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="gec5.reason",
                action="ACCEPT_QUOTE",
                decision=PolicyAction.ALLOW,
                reason="custom_allow_reason",
            ),
        )
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.reason == "custom_allow_reason"


@pytest.mark.unit
@pytest.mark.gate
def test_policy_engine_facade_delegates() -> None:
    facade = PolicyEngine(
        runtime=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="gec5.facade",
                    action="CREATE_EXTERNAL_WORK",
                    decision=PolicyAction.ALLOW,
                ),
            )
        )
    )
    decision = facade.evaluate_meaningful_side_effect(
        _request(action="CREATE_EXTERNAL_WORK", kinds=(MeaningfulSideEffectKind.MUTATION,))
    )
    assert decision.action is PolicyAction.ALLOW
