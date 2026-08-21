# © Artur Czarnecki. All rights reserved.

"""G5C-2B-2B — fresh re-evaluation + single-use governed grant consumption."""

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
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_OPERATION = "collaborative.document.delete"
_OPERATION_OTHER = "collaborative.document.publish"
_SCOPE = "document.delete"
_RESOURCE = "document-123"
_RESOURCE_OTHER = "document-456"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_TASK_ID = mint_task_id()
_RUN_ID = "run-g5c2b2b-1"
_RUN_OTHER = "run-g5c2b2b-2"
_POLICY_RULE = "runtime.hitl"
_BUNDLE_ID = "bundle-g5c2b2b"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_V2 = "2.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_BUNDLE_D2 = "sha256:" + ("22" * 32)
_SCOPE_1 = "side-effect-scope-1"
_SCOPE_2 = "side-effect-scope-2"
_SCOPE_DIGEST_1 = "sha256:" + ("ab" * 32)
_SCOPE_DIGEST_2 = "sha256:" + ("cd" * 32)


class MutableRuntimePolicyEvaluator:
    """Deterministic evaluator for fresh re-evaluation proof tests."""

    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision
        self.calls: list[MeaningfulSideEffectRequest] = []

    def set_decision(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        self.calls.append(request)
        return self._decision


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
        reason="test",
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
    side_effect_scope_digest: str | None = None,
    policy_bundle_version: str = _BUNDLE_V1,
    policy_bundle_digest: str = _BUNDLE_D1,
) -> GovernedContinuationApprovalGrant:
    return GovernedContinuationApprovalGrant(
        grant_id="gcg_g5c2b2b_test",
        continuation_request_id="gcr_g5c2b2b_test",
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        task_id=task_id,
        run_id=run_id,
        operation_id=operation_id,
        resource_scope=resource_scope,
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
        pause_id="pause-g5c2b2b",
        human_request_id="hr-g5c2b2b",
        approved_at="2026-08-19T00:00:00+00:00",
    )


def _task(task_id: str = _TASK_ID) -> Task:
    return Task(tenant_id="t1", user_id="u1", message="x", task_id=task_id)


def _lifecycle(task: Task) -> TaskLifecycle:
    lifecycle = TaskLifecycle()
    lifecycle.transition(task, TaskState.CLASSIFIED)
    lifecycle.transition(task, TaskState.PLANNED)
    return lifecycle


def _seed_gate(
    *,
    policy_evaluator: MutableRuntimePolicyEvaluator,
    operation_id: str = _OPERATION,
    resource_scope: str = _RESOURCE,
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
        runtime_policy_evaluator=policy_evaluator,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate), membership


def _enforcement_request(
    membership: WorkspaceMembership,
    *,
    operation_id: str = _OPERATION,
    resource_scope: str = _RESOURCE,
    task_id: str = _TASK_ID,
    run_id: str = _RUN_ID,
    side_effect_scope_id: str = _SCOPE_1,
    side_effect_scope_digest: str | None = None,
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
            side_effect_scope_id=side_effect_scope_id,
            side_effect_scope_digest=side_effect_scope_digest,
            task_id=task_id,
            run_id=run_id,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=resource_scope,
        ),
    )


def _run_boundary(
    boundary: MeaningfulSideEffectAuthorizationBoundary,
    membership: WorkspaceMembership,
    *,
    task: Task | None = None,
    lifecycle: TaskLifecycle | None = None,
    counter: list[int],
    raises: bool = False,
    **request_kwargs: object,
) -> object:
    def _execute() -> str:
        counter[0] += 1
        if raises:
            raise RuntimeError("execute_failed_after_consumption")
        return "ok"

    return boundary.authorize_and_execute(
        _enforcement_request(membership, **request_kwargs),
        _execute,
        task=task,
        lifecycle=lifecycle,
    )


def test_exact_match_executes_once_and_consumes_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert result == "ok"
    assert counter[0] == 1
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.runtime.governance.paused is False


def test_second_use_enters_new_hitl_without_execution() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    first = _run_boundary(boundary, membership, task=task, counter=counter)
    assert first == "ok"
    assert counter[0] == 1

    lifecycle.transition(task, TaskState.RUNNING)
    second = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
    )
    assert isinstance(second, MeaningfulSideEffectAuthorizationResult)
    assert second.decision.action is PolicyAction.REQUIRE_HUMAN
    assert counter[0] == 1
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN
    assert task.runtime.governance.paused is True


def test_deny_after_approval_blocks_execution_and_clears_obsolete_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.DENY))
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None


def test_allow_after_approval_executes_and_clears_obsolete_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ALLOW))
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert result == "ok"
    assert counter[0] == 1
    assert task.runtime.governance.governed_continuation_grant is None


def test_policy_version_changed_clears_grant_and_enters_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(
        _decision(policy_bundle_version=_BUNDLE_V2, policy_bundle_digest=_BUNDLE_D2)
    )
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_policy_digest_changed_enters_new_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(
        _decision(policy_bundle_digest=_BUNDLE_D2)
    )
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_side_effect_scope_s1_to_s2_no_execute_new_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant(side_effect_scope_id=_SCOPE_1)
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
        side_effect_scope_id=_SCOPE_2,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_material_digest_changed_no_execute_new_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant(
        side_effect_scope_digest=_SCOPE_DIGEST_1
    )
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
        side_effect_scope_digest=_SCOPE_DIGEST_2,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_run_mismatch_no_execute() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant(run_id=_RUN_ID)
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
        run_id=_RUN_OTHER,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None


def test_operation_mismatch_no_execute() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(
        policy_evaluator=evaluator,
        operation_id=_OPERATION_OTHER,
        resource_scope=_RESOURCE_OTHER,
    )
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
        operation_id=_OPERATION_OTHER,
        resource_scope=_RESOURCE_OTHER,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0


def test_resource_mismatch_no_execute() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(
        policy_evaluator=evaluator,
        resource_scope=_RESOURCE_OTHER,
    )
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant(resource_scope=_RESOURCE)
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
        resource_scope=_RESOURCE_OTHER,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0


def test_execute_raises_after_consumption_grant_stays_consumed() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    with pytest.raises(RuntimeError, match="execute_failed_after_consumption"):
        _run_boundary(
            boundary,
            membership,
            task=task,
            counter=counter,
            raises=True,
        )
    assert counter[0] == 1
    assert task.runtime.governance.governed_continuation_grant is None

    lifecycle = _lifecycle(task)
    retry = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
    )
    assert isinstance(retry, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 1
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_require_human_without_task_no_execute() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    counter = [0]

    result = _run_boundary(boundary, membership, counter=counter)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0


def test_require_human_without_lifecycle_and_no_grant_fail_closed() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    counter = [0]

    result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.paused is False


def test_escalate_never_executes_with_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ESCALATE))
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.ESCALATE
    assert counter[0] == 0


def test_non_bundle_require_human_no_execute_new_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(
        PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="no_bundle",
            policy_rule_id=_POLICY_RULE,
        )
    )
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    lifecycle = _lifecycle(task)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    result = _run_boundary(
        boundary,
        membership,
        task=task,
        lifecycle=lifecycle,
        counter=counter,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert counter[0] == 0
    assert task.runtime.governance.governed_continuation_grant is None
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_fresh_evaluation_uses_current_policy_not_stored_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_gate(policy_evaluator=evaluator)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    _run_boundary(boundary, membership, task=task, counter=counter)
    assert len(evaluator.calls) == 1

    evaluator.set_decision(_decision(action=PolicyAction.DENY))
    task.runtime.governance.governed_continuation_grant = _grant()
    deny_result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert isinstance(deny_result, MeaningfulSideEffectAuthorizationResult)
    assert deny_result.decision.action is PolicyAction.DENY
    assert counter[0] == 1
    assert len(evaluator.calls) == 2

    evaluator.set_decision(_decision(action=PolicyAction.ALLOW))
    task.runtime.governance.governed_continuation_grant = _grant()
    allow_result = _run_boundary(boundary, membership, task=task, counter=counter)
    assert allow_result == "ok"
    assert counter[0] == 2
