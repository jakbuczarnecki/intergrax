# © Artur Czarnecki. All rights reserved.

"""G5C-2B-2A — exact policy requirement provenance + fail-closed grant matcher."""

from __future__ import annotations

import pytest

from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.human.governed_continuation_grant import matches_current_requirement

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = "task-matcher-1"
RUN_ID = "run-matcher-1"
RUN_OTHER = "run-matcher-2"
TASK_OTHER = "task-matcher-2"
OPERATION = "collaborative.document.delete"
OPERATION_OTHER = "collaborative.document.publish"
RESOURCE = "document-123"
RESOURCE_OTHER = "document-456"
POLICY_RULE = "runtime.hitl"
POLICY_RULE_OTHER = "runtime.deny"
SCOPE_1 = "side-effect-scope-1"
SCOPE_2 = "side-effect-scope-2"
SCOPE_DIGEST_1 = "sha256:" + ("ab" * 32)
SCOPE_DIGEST_2 = "sha256:" + ("cd" * 32)
BUNDLE_ID = "bundle-test"
BUNDLE_V1 = "1.0.0"
BUNDLE_V2 = "2.0.0"
BUNDLE_D1 = "sha256:" + ("11" * 32)
BUNDLE_D2 = "sha256:" + ("22" * 32)


def _side_effect(
    *,
    task_id: str = TASK_ID,
    run_id: str = RUN_ID,
    side_effect_scope_id: str = SCOPE_1,
    side_effect_scope_digest: str | None = None,
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action="DELETE_DOCUMENT",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        task_id=task_id,
        run_id=run_id,
    )


def _grant(
    *,
    task_id: str = TASK_ID,
    run_id: str = RUN_ID,
    operation_id: str = OPERATION,
    resource_scope: str | None = RESOURCE,
    side_effect_scope_id: str = SCOPE_1,
    side_effect_scope_digest: str | None = None,
    policy_rule_id: str = POLICY_RULE,
    policy_bundle_id: str = BUNDLE_ID,
    policy_bundle_version: str = BUNDLE_V1,
    policy_bundle_digest: str = BUNDLE_D1,
) -> GovernedContinuationApprovalGrant:
    return GovernedContinuationApprovalGrant(
        grant_id="gcg_matcher_test",
        continuation_request_id="gcr_matcher_test",
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        task_id=task_id,
        run_id=run_id,
        operation_id=operation_id,
        resource_scope=resource_scope,
        policy_rule_id=policy_rule_id,
        policy_bundle_id=policy_bundle_id,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
        pause_id="pause-matcher",
        human_request_id="hr-matcher",
        approved_at="2026-08-19T00:00:00+00:00",
    )


def _decision(
    *,
    action: PolicyAction = PolicyAction.REQUIRE_HUMAN,
    policy_rule_id: str = POLICY_RULE,
    policy_bundle_id: str = BUNDLE_ID,
    policy_bundle_version: str = BUNDLE_V1,
    policy_bundle_digest: str = BUNDLE_D1,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="test",
        policy_rule_id=policy_rule_id,
        policy_bundle_id=policy_bundle_id,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
    )


def _match(
    grant: GovernedContinuationApprovalGrant,
    *,
    side_effect: MeaningfulSideEffectRequest | None = None,
    operation_id: str = OPERATION,
    resource_scope: str | None = RESOURCE,
    decision: PolicyDecision | None = None,
) -> bool:
    return matches_current_requirement(
        grant,
        current_side_effect=side_effect or _side_effect(),
        current_operation_id=operation_id,
        current_resource_scope=resource_scope,
        current_decision=decision or _decision(),
    )


def test_exact_requirement_match() -> None:
    grant = _grant()
    assert _match(grant) is True


def test_policy_bundle_version_change_no_match() -> None:
    grant = _grant()
    decision = _decision(policy_bundle_version=BUNDLE_V2, policy_bundle_digest=BUNDLE_D2)
    assert _match(grant, decision=decision) is False


def test_policy_bundle_digest_change_no_match() -> None:
    grant = _grant()
    decision = _decision(policy_bundle_digest=BUNDLE_D2)
    assert _match(grant, decision=decision) is False


def test_policy_rule_change_no_match() -> None:
    grant = _grant()
    decision = _decision(policy_rule_id=POLICY_RULE_OTHER)
    assert _match(grant, decision=decision) is False


def test_side_effect_scope_id_mismatch_no_match() -> None:
    grant = _grant()
    side_effect = _side_effect(side_effect_scope_id=SCOPE_2)
    assert _match(grant, side_effect=side_effect) is False


def test_side_effect_scope_digest_mismatch_no_match() -> None:
    grant = _grant(side_effect_scope_digest=SCOPE_DIGEST_1)
    side_effect = _side_effect(side_effect_scope_digest=SCOPE_DIGEST_2)
    assert _match(grant, side_effect=side_effect) is False


def test_side_effect_scope_digest_one_none_no_match() -> None:
    grant = _grant(side_effect_scope_digest=SCOPE_DIGEST_1)
    side_effect = _side_effect(side_effect_scope_digest=None)
    assert _match(grant, side_effect=side_effect) is False


def test_side_effect_scope_digest_both_none_match() -> None:
    grant = _grant(side_effect_scope_digest=None)
    side_effect = _side_effect(side_effect_scope_digest=None)
    assert _match(grant, side_effect=side_effect) is True


def test_task_mismatch_no_match() -> None:
    grant = _grant()
    side_effect = _side_effect(task_id=TASK_OTHER)
    assert _match(grant, side_effect=side_effect) is False


def test_run_mismatch_no_match() -> None:
    grant = _grant()
    side_effect = _side_effect(run_id=RUN_OTHER)
    assert _match(grant, side_effect=side_effect) is False


def test_operation_mismatch_no_match() -> None:
    grant = _grant()
    assert _match(grant, operation_id=OPERATION_OTHER) is False


def test_resource_scope_mismatch_no_match() -> None:
    grant = _grant()
    assert _match(grant, resource_scope=RESOURCE_OTHER) is False


def test_current_deny_no_match() -> None:
    grant = _grant()
    decision = _decision(action=PolicyAction.DENY)
    assert _match(grant, decision=decision) is False


def test_current_allow_no_match() -> None:
    grant = _grant()
    decision = _decision(action=PolicyAction.ALLOW)
    assert _match(grant, decision=decision) is False


def test_current_escalate_no_match() -> None:
    grant = _grant()
    decision = _decision(action=PolicyAction.ESCALATE)
    assert _match(grant, decision=decision) is False


def test_non_bundle_require_human_no_match() -> None:
    grant = _grant(
        policy_bundle_id="",
        policy_bundle_version="",
        policy_bundle_digest="",
    )
    decision = PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="test",
        policy_rule_id=POLICY_RULE,
    )
    assert _match(grant, decision=decision) is False


def test_grant_without_bundle_current_with_bundle_no_match() -> None:
    grant = _grant(
        policy_bundle_id="",
        policy_bundle_version="",
        policy_bundle_digest="",
    )
    assert _match(grant) is False


def test_grant_with_bundle_current_without_bundle_no_match() -> None:
    grant = _grant()
    decision = PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="test",
        policy_rule_id=POLICY_RULE,
    )
    assert _match(grant, decision=decision) is False


def test_matcher_is_pure_does_not_mutate_grant() -> None:
    grant = _grant()
    before = grant.model_copy()
    assert _match(grant) is True
    assert grant == before
