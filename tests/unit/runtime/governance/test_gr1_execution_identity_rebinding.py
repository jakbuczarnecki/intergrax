# © Artur Czarnecki. All rights reserved.

"""GR-1 — governed continuation authorization bound to canonical execution identity."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.governed_continuation_bridge import (
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.governed_continuation_bridge import (
    bridge_governed_continuation_to_execution_result,
)
from intergrax.runtime.task.task import Task
from intergrax.contracts.runtime_policy import PolicyDecision

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
ATTEMPT_ID = mint_attempt_id()
EXECUTION_ID = mint_execution_id()
CHILD_EXECUTION_ID = mint_execution_id()
OPERATION = "collaborative.document.delete"
SCOPE = "scope-gr1-1"
APPROVER = local_development_approver_evidence(tenant_id="t1")


def _side_effect(
    *,
    attempt_id: str = ATTEMPT_ID,
    execution_id: str = EXECUTION_ID,
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action="DELETE",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=SCOPE,
        task_id=TASK_ID,
        run_id=RUN_ID,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def _decision() -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="gr1",
        policy_rule_id="runtime.hitl",
        policy_bundle_id="bundle-gr1",
        policy_bundle_version="1.0.0",
        policy_bundle_digest="sha256:" + ("aa" * 32),
    )


def test_full_typed_identity_chain_to_grant() -> None:
    side_effect = _side_effect()
    enforcement = CollaborativeWorkEnforcementRequest(
        tenant_id="t1",
        workspace_id="ws1",
        operation_id=OPERATION,
        acting_principal_id="principal-1",
        resource_scope="doc-1",
        meaningful_side_effect_request=side_effect,
    )
    continuation = compose_governed_continuation_from_enforcement(
        enforcement,
        decision=_decision(),
        enforcement_operation_id=OPERATION,
        enforcement_authority_scope="doc-1",
        requires_governed_continuation=True,
        source_agent_id="agent-gr1",
    )
    assert continuation is not None
    assert continuation.attempt_id == ATTEMPT_ID
    assert continuation.execution_id == EXECUTION_ID

    correlation = continuation.to_correlation()
    assert correlation.attempt_id == ATTEMPT_ID
    assert correlation.execution_id == EXECUTION_ID

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    execution = bridge_governed_continuation_to_execution_result(continuation)
    HumanPauseCoordinator.apply_pause(task, execution)
    pause = task.runtime.governance.pause_record
    assert pause is not None
    assert task.runtime.governance.human_request is not None
    assert task.runtime.governance.human_request.governed_continuation == correlation

    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=APPROVER,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=EXECUTION_ID,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    assert grant.attempt_id == ATTEMPT_ID
    assert grant.execution_id == EXECUTION_ID


def test_child_execution_id_does_not_match_parent_grant() -> None:
    from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
    from intergrax.runtime.human.governed_continuation_grant import matches_current_requirement

    grant = GovernedContinuationApprovalGrant(
        grant_id="gcg_parent",
        continuation_request_id="gcr_parent",
        side_effect_scope_id=SCOPE,
        task_id=TASK_ID,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=EXECUTION_ID,
        operation_id=OPERATION,
        resource_scope="doc-1",
        policy_rule_id="runtime.hitl",
        policy_bundle_id="bundle-gr1",
        policy_bundle_version="1.0.0",
        policy_bundle_digest="sha256:" + ("aa" * 32),
        pause_id="pause-gr1",
        human_request_id="hr-gr1",
        approved_at="2026-09-14T00:00:00+00:00",
    )
    child_side_effect = _side_effect(execution_id=CHILD_EXECUTION_ID)
    assert (
        matches_current_requirement(
            grant,
            current_side_effect=child_side_effect,
            current_operation_id=OPERATION,
            current_resource_scope="doc-1",
            current_decision=_decision(),
        )
        is False
    )
