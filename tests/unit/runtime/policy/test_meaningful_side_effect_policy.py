# © Artur Czarnecki. All rights reserved.

"""GEC-5 — platform meaningful side-effect policy evaluation."""

from __future__ import annotations

import pytest

from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


def _request(**overrides: object) -> MeaningfulSideEffectRequest:
    payload: dict[str, object] = {
        "action": "ACCEPT_QUOTE",
        "kinds": (MeaningfulSideEffectKind.COMMITMENT, MeaningfulSideEffectKind.MUTATION),
        "task_id": "task-1",
        "run_id": "run-1",
        "principal_id": "user-1",
        "tenant_id": "tenant-a",
        "external_target": "provider-x",
    }
    payload.update(overrides)
    return MeaningfulSideEffectRequest.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_default_fail_closed_without_matching_rule() -> None:
    engine = RuntimePolicyEngine(rules=[])
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_indeterminate"


@pytest.mark.unit
@pytest.mark.gate
def test_missing_principal_fail_closed() -> None:
    engine = RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": "ACCEPT_QUOTE",
                "decision": "allow",
            }
        ]
    )
    decision = engine.evaluate_meaningful_side_effect(_request(principal_id=None))
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_principal_missing"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_allow_deny_require_human() -> None:
    allow_engine = RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": "ACCEPT_QUOTE",
                "decision": "allow",
                "id": "gec5.allow",
            }
        ]
    )
    assert (
        allow_engine.evaluate_meaningful_side_effect(_request()).action
        is PolicyAction.ALLOW
    )

    deny_engine = RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": "ACCEPT_QUOTE",
                "decision": "deny",
            }
        ]
    )
    assert (
        deny_engine.evaluate_meaningful_side_effect(_request()).action
        is PolicyAction.DENY
    )

    human_engine = RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": "ACCEPT_QUOTE",
                "decision": "require_human",
            }
        ]
    )
    assert (
        human_engine.evaluate_meaningful_side_effect(_request()).action
        is PolicyAction.REQUIRE_HUMAN
    )


@pytest.mark.unit
@pytest.mark.gate
def test_modify_unsupported_fail_closed() -> None:
    engine = RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": "ACCEPT_QUOTE",
                "decision": "modify",
            }
        ]
    )
    decision = engine.evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_unsupported_decision"


@pytest.mark.unit
@pytest.mark.gate
def test_policy_engine_facade_delegates() -> None:
    facade = PolicyEngine(
        runtime=RuntimePolicyEngine(
            rules=[
                {
                    "type": "meaningful_side_effect",
                    "action": "CREATE_EXTERNAL_WORK",
                    "decision": "allow",
                }
            ]
        )
    )
    decision = facade.evaluate_meaningful_side_effect(
        _request(action="CREATE_EXTERNAL_WORK", kinds=(MeaningfulSideEffectKind.MUTATION,))
    )
    assert decision.action is PolicyAction.ALLOW
