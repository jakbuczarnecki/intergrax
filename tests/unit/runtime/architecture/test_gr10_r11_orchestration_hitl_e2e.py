# © Artur Czarnecki. All rights reserved.

"""GR-10-R11 — orchestration HITL behavioral E2E proofs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    CollaborativeWorkEnforcementRequest,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotContinuationExecutor,
    GovernedOrchestrationSlotExecutor,
    OrchestrationConsequentialEffectBlockedError,
    authorize_orchestration_consequential_effect,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.mse_hitl_effect_gate import (
    MseHitlEffectGateDisposition,
    evaluate_mse_hitl_effect_gate,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_inner_guard,
)
from tests.unit.runtime.policy.test_g5c2b2b_governed_side_effect_reauthorization import (
    MutableRuntimePolicyEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-r11"
_WORKSPACE = "workspace-r11"
_ACTING = "principal-r11"
_OPERATION = "orch.hitl.effect"
_RESOURCE = "resource-r11"
_NOW = datetime(2026, 9, 19, 16, 0, tzinfo=UTC)
_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_ATTEMPT_ID = mint_attempt_id()
_EXECUTION_ID = mint_execution_id()
_POLICY_RULE = "runtime.hitl.r11"
_BUNDLE_ID = "bundle-r11"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_SCOPE_1 = "side-effect-scope-r11-a"
_SCOPE_2 = "side-effect-scope-r11-b"
_SCOPE_DIGEST_1 = "sha256:" + ("ab" * 32)


def _decision(
    *,
    action: PolicyAction = PolicyAction.REQUIRE_HUMAN,
    policy_rule_id: str = _POLICY_RULE,
    policy_bundle_id: str = _BUNDLE_ID,
    policy_bundle_version: str = _BUNDLE_V1,
    policy_bundle_digest: str = _BUNDLE_D1,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="r11-test",
        policy_rule_id=policy_rule_id,
        policy_bundle_id=policy_bundle_id,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
    )


def _grant(
    *,
    task_id: str = _TASK_ID,
    run_id: str = _RUN_ID,
    operation_id: str = _OPERATION,
    resource_scope: str | None = _RESOURCE,
    side_effect_scope_id: str = _SCOPE_1,
    side_effect_scope_digest: str | None = _SCOPE_DIGEST_1,
) -> GovernedContinuationApprovalGrant:
    return GovernedContinuationApprovalGrant(
        grant_id="gcg_r11_test",
        continuation_request_id="gcr_r11_test",
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        task_id=task_id,
        run_id=run_id,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        operation_id=operation_id,
        resource_scope=resource_scope,
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
        pause_id="pause-r11",
        human_request_id="hr-r11",
        approved_at="2026-09-19T00:00:00+00:00",
    )


def _seed_boundary(
    evaluator: MutableRuntimePolicyEvaluator,
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, WorkspaceMembership]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-r11",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-r11",
            principal_id=_ACTING,
            authority_scopes=("document.delete",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-r11",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope="document.delete",
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="resource-allow-r11",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope="document.delete",
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            authority_scope="document.delete",
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.REQUIRED,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=evaluator,
    )
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=default_gr3_inner_guard(_TASK_ID),
    ), membership


def _enforcement_request(
    membership: WorkspaceMembership,
    *,
    side_effect_scope_id: str = _SCOPE_1,
    side_effect_scope_digest: str | None = _SCOPE_DIGEST_1,
    task_id: str = _TASK_ID,
    run_id: str = _RUN_ID,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id=side_effect_scope_id,
            side_effect_scope_digest=side_effect_scope_digest,
            task_id=task_id,
            run_id=run_id,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )


def test_scenario_a_require_human_zero_effect_and_continuation_request() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request, source_agent_id="r11")
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=None,
        )
    assert authorization.decision.action is PolicyAction.REQUIRE_HUMAN
    assert authorization.requires_governed_continuation is True
    assert authorization.governed_continuation_request is not None
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert len(evaluator.calls) == 1


def test_scenario_b_approval_plus_allow_executes_once() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    def _execute() -> str:
        counter[0] += 1
        return "ok"

    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        # Post-human: fresh policy becomes ALLOW
        evaluator.set_decision(_decision(action=PolicyAction.ALLOW))
        result = boundary.authorize_and_execute(
            _enforcement_request(membership),
            _execute,
            task=task,
        )
    assert result == "ok"
    assert counter[0] == 1
    assert len(evaluator.calls) == 1


def test_scenario_c_approval_plus_deny_zero_effect() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.DENY))
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        result = boundary.authorize_and_execute(
            _enforcement_request(membership),
            lambda: counter.__setitem__(0, counter[0] + 1) or "ok",
            task=task,
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert counter[0] == 0


def test_scenario_d_approval_plus_require_human_again_pauses() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    lifecycle = TaskLifecycle()
    lifecycle.transition(task, TaskState.CLASSIFIED)
    lifecycle.transition(task, TaskState.PLANNED)
    lifecycle.transition(task, TaskState.RUNNING)
    counter = [0]

    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        result = boundary.authorize_and_execute(
            _enforcement_request(membership),
            lambda: counter.__setitem__(0, counter[0] + 1) or "ok",
            task=task,
            lifecycle=lifecycle,
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.governed_continuation_request is not None
    assert counter[0] == 0
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_scenario_e_rejected_blocks_via_missing_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    # REJECTED → no grant minted
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL


def test_scenario_f_stale_approval_wrong_scope_blocked() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant(
        side_effect_scope_id=_SCOPE_2,
    )
    request = _enforcement_request(membership, side_effect_scope_id=_SCOPE_1)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    # Stale wrong-scope grant is not consumed as authority for this proposal
    assert task.runtime.governance.governed_continuation_grant is not None
    assert (
        task.runtime.governance.governed_continuation_grant.side_effect_scope_id
        == _SCOPE_2
    )


def test_scenario_g_wrong_run_lineage_blocked() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    other_run = mint_run_id()
    task.runtime.governance.governed_continuation_grant = _grant(run_id=other_run)
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL


def test_scenario_h_cross_slot_scope_reuse_blocked() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant(
        side_effect_scope_id=_SCOPE_1,
    )
    request_b = _enforcement_request(membership, side_effect_scope_id=_SCOPE_2)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request_b)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request_b,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL


def test_scenario_i_missing_continuation_authority_zero_effect() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    # human-looking approval absent as grant
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        with pytest.raises(OrchestrationConsequentialEffectBlockedError) as exc_info:
            authorize_orchestration_consequential_effect(
                boundary,
                enforcement_request=request,
                production_mode=True,
                source_agent_id="r11",
                source_step_id="slot-a",
            )
    assert exc_info.value.governed_continuation_request is not None


@pytest.mark.asyncio
async def test_scenario_b_topology_continue_slot_grant_then_allow() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    effects = [0]

    class _Inner:
        async def continue_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "done"

    executor = GovernedOrchestrationSlotContinuationExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _slot, _payload: _enforcement_request(membership),
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(task)
    try:
        with bound_gr3_active_execution(
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        ):
            # First: REQUIRE_HUMAN without grant → blocked
            with pytest.raises(Exception):
                await executor.continue_slot(
                    slot_id=OrchestrationSlotId("slot-a"),
                    payload=object(),
                )
            assert effects[0] == 0

            # Fresh ALLOW after human → effect
            evaluator.set_decision(_decision(action=PolicyAction.ALLOW))
            result = await executor.continue_slot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "done"
    assert effects[0] == 1
    assert len(evaluator.calls) == 2


@pytest.mark.asyncio
async def test_scenario_c_topology_submit_require_human_zero_calls() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    effects = [0]

    class _Inner:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "done"

    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _slot, _payload: _enforcement_request(membership),
    )
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        with pytest.raises(Exception):
            await executor.execute_slot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=object(),
            )
    assert effects[0] == 0


def test_scenario_k_custom_runtime_policy_evaluator_replaceable() -> None:
    """Protocol-shaped custom evaluator — platform gate unchanged."""
    custom = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ALLOW))
    boundary, membership = _seed_boundary(custom)
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=None,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.PROCEED
    assert isinstance(custom, MutableRuntimePolicyEvaluator)


def test_freshness_initial_and_post_human_evaluations() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        first = boundary.authorize(request)
        assert first.decision.action is PolicyAction.REQUIRE_HUMAN
        assert len(evaluator.calls) == 1
        evaluator.set_decision(_decision(action=PolicyAction.DENY))
        second = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            second,
            enforcement_request=request,
            task=task,
        )
    assert second.decision.action is PolicyAction.DENY
    assert gate.disposition is MseHitlEffectGateDisposition.BLOCK
    assert len(evaluator.calls) == 2


def test_grant_match_require_human_never_proceeds() -> None:
    """R11-R1 Scenario D — matching grant cannot override fresh REQUIRE_HUMAN."""
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert task.runtime.governance.governed_continuation_grant is None


def test_escalate_never_proceeds_with_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ESCALATE))
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert authorization.requires_governed_continuation is True
