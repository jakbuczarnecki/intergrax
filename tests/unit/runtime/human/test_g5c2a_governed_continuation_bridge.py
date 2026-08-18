# © Artur Czarnecki. All rights reserved.

"""G5C-2A — governed continuation bridge into canonical HITL lifecycle."""

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
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
    bridge_governed_continuation_to_execution_result,
    bridge_governed_continuation_to_governance,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_OPERATION_S1 = "collaborative.document.delete"
_OPERATION_S2 = "collaborative.document.publish"
_SCOPE = "document.delete"
_RESOURCE_S1 = "document-123"
_RESOURCE_S2 = "document-456"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _task(task_id: str | None = None) -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="x",
        task_id=task_id or mint_task_id(),
    )


def _seed_gate(
    *,
    runtime_policy: RuntimePolicyEngine | None = None,
    operation_id: str = _OPERATION_S1,
    resource_scope: str = _RESOURCE_S1,
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, WorkspaceMembership]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="resource-allow",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=resource_scope,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=operation_id,
            authority_scope=_SCOPE,
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
        runtime_policy_evaluator=runtime_policy
        or RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.allow",
                    action=operation_id,
                    decision=PolicyAction.ALLOW,
                ),
            )
        ),
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate), membership


def _enforcement_request(
    membership: WorkspaceMembership,
    *,
    operation_id: str = _OPERATION_S1,
    resource_scope: str = _RESOURCE_S1,
    task_id: str = "task-1",
    run_id: str = "run-1",
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=operation_id,
        acting_principal_id=_ACTING,
        resource_scope=resource_scope,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=operation_id,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            task_id=task_id,
            run_id=run_id,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=resource_scope,
        ),
    )


def _continuation_request(
    *,
    operation_id: str = _OPERATION_S1,
    resource_scope: str = _RESOURCE_S1,
    continuation_request_id: str = "gcr_test_1",
) -> GovernedContinuationRequest:
    return GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id="task-1",
        run_id="run-1",
        source_agent_id="agent-test",
        prompt="continuation required",
        continuation_request_id=continuation_request_id,
        operation_id=operation_id,
        policy_rule_id="runtime.hitl",
        resource_scope=resource_scope,
        policy_action=PolicyAction.REQUIRE_HUMAN,
    )


def test_require_human_produces_canonical_pause_composition() -> None:
    boundary, membership = _seed_gate(
        runtime_policy=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.hitl",
                    action=_OPERATION_S1,
                    decision=PolicyAction.REQUIRE_HUMAN,
                ),
            )
        )
    )
    executed: list[str] = []
    authorization = boundary.authorize(_enforcement_request(membership))
    assert authorization.permitted is False
    assert authorization.requires_governed_continuation is True
    assert authorization.governed_continuation_request is not None

    result = boundary.authorize_and_execute(
        _enforcement_request(membership),
        lambda: executed.append("side-effect"),
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert executed == []

    task = _task()
    lifecycle = TaskLifecycle()
    lifecycle.transition(task, TaskState.CLASSIFIED)
    lifecycle.transition(task, TaskState.PLANNED)

    continuation = authorization.governed_continuation_request
    apply_governed_continuation_pause(task, continuation)
    lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)

    gov = task.runtime.governance
    assert gov.paused is True
    assert gov.pause_record is not None
    assert gov.human_request is not None
    assert gov.execution_interrupt is not None
    assert gov.human_request.governed_continuation is not None
    assert (
        gov.human_request.governed_continuation.continuation_request_id
        == continuation.continuation_request_id
    )


def test_exact_continuation_correlation_distinguishes_s1_from_s2() -> None:
    continuation_s1 = _continuation_request(
        operation_id=_OPERATION_S1,
        resource_scope=_RESOURCE_S1,
        continuation_request_id="gcr_s1",
    )
    continuation_s2 = _continuation_request(
        operation_id=_OPERATION_S2,
        resource_scope=_RESOURCE_S2,
        continuation_request_id="gcr_s2",
    )

    resolution_s1 = bridge_governed_continuation_to_governance(continuation_s1)
    resolution_s2 = bridge_governed_continuation_to_governance(continuation_s2)

    corr_s1 = resolution_s1.human_request.governed_continuation
    corr_s2 = resolution_s2.human_request.governed_continuation
    assert corr_s1 is not None and corr_s2 is not None
    assert corr_s1.operation_id == _OPERATION_S1
    assert corr_s2.operation_id == _OPERATION_S2
    assert corr_s1.resource_scope == _RESOURCE_S1
    assert corr_s2.resource_scope == _RESOURCE_S2
    assert corr_s1.continuation_request_id != corr_s2.continuation_request_id


def test_deny_does_not_create_hitl_or_continuation() -> None:
    boundary, membership = _seed_gate(
        runtime_policy=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.deny",
                    action=_OPERATION_S1,
                    decision=PolicyAction.DENY,
                ),
            )
        )
    )
    authorization = boundary.authorize(_enforcement_request(membership))
    assert authorization.permitted is False
    assert authorization.requires_governed_continuation is False
    assert authorization.governed_continuation_request is None

    task = _task()
    assert task.runtime.governance.paused is False
    assert task.runtime.governance.pause_record is None
    assert task.runtime.governance.human_request is None
    assert task.runtime.governance.execution_interrupt is None


def test_allow_does_not_pause() -> None:
    boundary, membership = _seed_gate()
    authorization = boundary.authorize(_enforcement_request(membership))
    assert authorization.permitted is True
    assert authorization.requires_governed_continuation is False
    assert authorization.governed_continuation_request is None

    executed: list[str] = []
    result = boundary.authorize_and_execute(
        _enforcement_request(membership),
        lambda: executed.append("ok") or "ok",
    )
    assert result == "ok"
    assert executed == ["ok"]


def test_continuation_request_is_not_execution_authority() -> None:
    continuation = _continuation_request()
    execution = bridge_governed_continuation_to_execution_result(continuation)
    assert execution.status.value == "needs_input"
    assert execution.human_request is not None

    boundary, membership = _seed_gate()
    executed: list[str] = []
    result = boundary.authorize_and_execute(
        _enforcement_request(membership),
        lambda: executed.append("ok") or "ok",
    )
    assert result == "ok"
    assert executed == ["ok"]

    deny_boundary, deny_membership = _seed_gate(
        runtime_policy=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.hitl",
                    action=_OPERATION_S1,
                    decision=PolicyAction.REQUIRE_HUMAN,
                ),
            )
        )
    )
    blocked = deny_boundary.authorize_and_execute(
        _enforcement_request(deny_membership),
        lambda: executed.append("blocked"),
    )
    assert isinstance(blocked, MeaningfulSideEffectAuthorizationResult)
    assert "blocked" not in executed


def test_canonical_hitl_reuse_human_pause_coordinator() -> None:
    task = _task()
    continuation = _continuation_request()
    execution = bridge_governed_continuation_to_execution_result(continuation)

    HumanPauseCoordinator.apply_pause(task, execution)

    gov = task.runtime.governance
    assert gov.paused is True
    assert gov.pause_record is not None
    assert gov.pause_record.human_request_id == execution.human_request.request_id
    assert gov.execution_interrupt is not None
    assert gov.execution_interrupt.interrupt_id == execution.execution_interrupt.interrupt_id
