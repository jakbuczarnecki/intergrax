# © Artur Czarnecki. All rights reserved.

"""GR-10-R14-R1 — post-HITL evidence correlation and root ESCALATE evidence proofs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.task.task import Task
from tests.unit.runtime.architecture.test_gr10_r14_orchestration_governance_evidence_e2e import (
    _OPERATION,
    _RESOURCE,
    _SCOPE,
    _TENANT,
    _WORKSPACE,
    _ACTING,
    _seeded_boundary,
)
from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MSE_AUTH = _REPO / "intergrax/runtime/policy/meaningful_side_effect_authorization.py"
_HUMAN_REF = "hr-r14r1-exact"
_SCOPE_A = "r14-side-effect-scope-a"
_SCOPE_B = "r14-side-effect-scope-b"
_SCOPE_DIGEST = "sha256:" + ("ab" * 32)


class _CapturingPersistence(GovernanceEvidencePersistencePort):
    def __init__(self) -> None:
        self.facts: list[GovernanceDecisionEvidenceFact] = []

    def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
        self.facts.append(fact)
        return GovernanceEvidencePersistenceOutcome(persisted=True, evidence_id=fact.evidence_id)


def _grant(identity: dict[str, object], **overrides: object) -> GovernedContinuationApprovalGrant:
    payload = {
        "grant_id": "gcg_r14r1",
        "continuation_request_id": "cont-r14r1",
        "side_effect_scope_id": _SCOPE_A,
        "side_effect_scope_digest": _SCOPE_DIGEST,
        **identity,
        "operation_id": _OPERATION,
        "resource_scope": _RESOURCE,
        "policy_rule_id": "r14.runtime.allow",
        "policy_bundle_id": "bundle-r14",
        "policy_bundle_version": "1",
        "policy_bundle_digest": "sha256:" + ("11" * 32),
        "pause_id": "pause-r14",
        "human_request_id": _HUMAN_REF,
        "approved_at": "2026-09-21T00:00:00+00:00",
    }
    payload.update(overrides)
    return GovernedContinuationApprovalGrant.model_validate(payload)


def _authorize_with_grant(
    store: _CapturingPersistence,
    grant_overrides: dict[str, object] | None,
    *,
    scope_id: str = _SCOPE_A,
) -> None:
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(store)
    side_effect = request.meaningful_side_effect_request
    assert side_effect is not None
    side_effect = side_effect.model_copy(
        update={
            "side_effect_scope_id": scope_id,
            "side_effect_scope_digest": _SCOPE_DIGEST,
        },
    )
    request = request.model_copy(update={"meaningful_side_effect_request": side_effect})
    host_task = Task(
        tenant_id=_TENANT,
        user_id=_ACTING,
        message="r14",
        task_id=side_effect.task_id,
    )
    if grant_overrides is not None:
        identity = {
            "task_id": side_effect.task_id,
            "run_id": side_effect.run_id,
            "attempt_id": side_effect.attempt_id,
            "execution_id": side_effect.execution_id,
        }
        host_task.runtime.governance.governed_continuation_grant = _grant(
            identity,
            **grant_overrides,
        )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(host_task)
    try:
        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ):
            boundary.authorize(request)
    finally:
        governed.reset(token)


def test_post_hitl_fresh_allow_correlates_exact_human_review_ref() -> None:
    store = _CapturingPersistence()
    _authorize_with_grant(store, {})
    assert len(store.facts) == 1
    assert store.facts[0].decision is PolicyAction.ALLOW
    assert store.facts[0].human_review_evidence_ref == _HUMAN_REF


def test_stale_grant_different_proposal_scope_has_no_human_ref() -> None:
    store = _CapturingPersistence()
    _authorize_with_grant(store, {"side_effect_scope_id": _SCOPE_B}, scope_id=_SCOPE_A)
    assert store.facts[0].human_review_evidence_ref is None


def test_cross_run_mismatch_has_no_human_ref() -> None:
    store = _CapturingPersistence()
    other_run = mint_run_id()
    _authorize_with_grant(store, {"run_id": other_run})
    assert store.facts[0].human_review_evidence_ref is None


def test_cross_resource_mismatch_has_no_human_ref() -> None:
    store = _CapturingPersistence()
    _authorize_with_grant(store, {"resource_scope": "other-resource"})
    assert store.facts[0].human_review_evidence_ref is None


def test_cross_side_effect_digest_mismatch_has_no_human_ref() -> None:
    store = _CapturingPersistence()
    _authorize_with_grant(
        store,
        {"side_effect_scope_digest": "sha256:" + ("99" * 32)},
    )
    assert store.facts[0].human_review_evidence_ref is None


def test_human_ref_correlation_does_not_change_governance_allow() -> None:
    store_with = _CapturingPersistence()
    store_without = _CapturingPersistence()
    _authorize_with_grant(store_with, {})
    _authorize_with_grant(store_without, None)
    assert store_with.facts[0].decision is PolicyAction.ALLOW
    assert store_without.facts[0].decision is PolicyAction.ALLOW
    assert store_with.facts[0].human_review_evidence_ref == _HUMAN_REF
    assert store_without.facts[0].human_review_evidence_ref is None


def test_fresh_allow_correlation_does_not_call_matches_current_requirement() -> None:
    source = _MSE_AUTH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_MSE_AUTH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_human_review_evidence_ref_for_allow":
            body = ast.unparse(node)
            assert "matches_current_requirement" not in body
            return
    pytest.fail("_resolve_human_review_evidence_ref_for_allow not found")
