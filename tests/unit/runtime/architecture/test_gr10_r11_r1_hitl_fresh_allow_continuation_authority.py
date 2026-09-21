# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R1 — fresh ALLOW + canonical continuation authority proofs."""

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
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    apply_resolution_to_pending,
    apply_resume_to_pending,
    assert_execution_continuation_identity_match,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
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
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.mse_hitl_effect_gate import (
    MseHitlEffectGateDisposition,
    evaluate_mse_hitl_effect_gate,
)
from intergrax.runtime.task.task import Task
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_inner_guard,
)
from tests.unit.runtime.policy.test_g5c2b2b_governed_side_effect_reauthorization import (
    MutableRuntimePolicyEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-r11r1"
_WORKSPACE = "workspace-r11r1"
_ACTING = "principal-r11r1"
_OPERATION = "orch.hitl.effect"
_RESOURCE = "resource-r11r1"
_NOW = datetime(2026, 9, 20, 7, 0, tzinfo=UTC)
_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_ATTEMPT_ID = mint_attempt_id()
_EXECUTION_ID = mint_execution_id()
_POLICY_RULE = "runtime.hitl.r11r1"
_BUNDLE_ID = "bundle-r11r1"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_BUNDLE_D2 = "sha256:" + ("22" * 32)
_SCOPE_1 = "side-effect-scope-r11r1-a"
_SCOPE_DIGEST_1 = "sha256:" + ("ab" * 32)
_CONTINUATION_ID = "gcr_r11r1_test"


def _decision(
    *,
    action: PolicyAction = PolicyAction.REQUIRE_HUMAN,
    policy_bundle_digest: str = _BUNDLE_D1,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="r11r1-test",
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=policy_bundle_digest,
    )


def _grant(**overrides: object) -> GovernedContinuationApprovalGrant:
    payload = {
        "grant_id": "gcg_r11r1_test",
        "continuation_request_id": _CONTINUATION_ID,
        "side_effect_scope_id": _SCOPE_1,
        "side_effect_scope_digest": _SCOPE_DIGEST_1,
        "task_id": _TASK_ID,
        "run_id": _RUN_ID,
        "attempt_id": _ATTEMPT_ID,
        "execution_id": _EXECUTION_ID,
        "operation_id": _OPERATION,
        "resource_scope": _RESOURCE,
        "policy_rule_id": _POLICY_RULE,
        "policy_bundle_id": _BUNDLE_ID,
        "policy_bundle_version": _BUNDLE_V1,
        "policy_bundle_digest": _BUNDLE_D1,
        "pause_id": "pause-r11r1",
        "human_request_id": "hr-r11r1",
        "approved_at": "2026-09-20T00:00:00+00:00",
    }
    payload.update(overrides)
    return GovernedContinuationApprovalGrant.model_validate(payload)


def _correlation(
    *,
    continuation_id: str = _CONTINUATION_ID,
) -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.COMPLIANCE,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        side_effect_scope_id=_SCOPE_1,
        side_effect_scope_digest=_SCOPE_DIGEST_1,
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
    )


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )


class _FakeContinuationPort:
    def __init__(self) -> None:
        self._store: dict[str, PendingExecutionContinuation] = {}

    def seed(
        self,
        *,
        continuation_id: str = _CONTINUATION_ID,
        state: ExecutionContinuationLifecycleState,
        identity: ExecutionContinuationIdentity | None = None,
        governed_correlation: GovernedContinuationCorrelation | None = ...,
    ) -> PendingExecutionContinuation:
        correlation: GovernedContinuationCorrelation | None
        if governed_correlation is ...:
            correlation = _correlation(continuation_id=continuation_id)
        else:
            correlation = governed_correlation
        pending = PendingExecutionContinuation(
            continuation_id=continuation_id,
            identity=identity or _identity(),
            lifecycle_state=state,
            revision=5,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=correlation,
            pause_id="pause-r11r1",
            human_request_id="hr-r11r1",
            requested_at="2026-09-20T00:00:00+00:00",
        )
        self._store[continuation_id] = pending
        return pending

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        raise ExecutionContinuationError(
            "not used",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        if lookup.continuation_id is not None:
            try:
                pending = self._store[lookup.continuation_id]
            except KeyError as exc:
                raise ExecutionContinuationError(
                    "continuation not found",
                    code=ExecutionContinuationErrorCode.NOT_FOUND,
                ) from exc
        else:
            matches = [
                p
                for p in self._store.values()
                if lookup.identity is not None
                and p.identity == lookup.identity
            ]
            if len(matches) != 1:
                raise ExecutionContinuationError(
                    "continuation not found",
                    code=ExecutionContinuationErrorCode.NOT_FOUND,
                )
            pending = matches[0]
        if lookup.identity is not None:
            assert_execution_continuation_identity_match(lookup.identity, pending.identity)
        return pending

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        pending = self.get_pending(
            ExecutionContinuationLookup(continuation_id=command.continuation_id)
        )
        updated = apply_resolution_to_pending(pending, command)
        self._store[command.continuation_id] = updated
        return updated

    def resume(self, command: ExecutionContinuationResumeCommand) -> PendingExecutionContinuation:
        pending = self.get_pending(
            ExecutionContinuationLookup(continuation_id=command.continuation_id)
        )
        updated = apply_resume_to_pending(pending, command)
        self._store[command.continuation_id] = updated
        return updated


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
            membership_id="membership-r11r1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-r11r1",
            principal_id=_ACTING,
            authority_scopes=("document.delete",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    for rule_id, layer, resource in (
        ("workspace-allow-r11r1", PolicyCompositionLayer.WORKSPACE_POLICY, None),
        ("resource-allow-r11r1", PolicyCompositionLayer.RESOURCE_POLICY, _RESOURCE),
    ):
        kwargs: dict[str, object] = {
            "tenant_id": _TENANT,
            "workspace_id": _WORKSPACE,
            "policy_rule_id": rule_id,
            "layer": layer,
            "authority_scope": "document.delete",
            "action": PolicyAction.ALLOW,
            "status": CollaborativePolicyRuleStatus.ACTIVE,
        }
        if resource is not None:
            kwargs["resource_scope"] = resource
        policy_repo.create(CreateCollaborativePolicyRuleCommand(**kwargs))
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


def _enforcement_request(membership: WorkspaceMembership) -> CollaborativeWorkEnforcementRequest:
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
            side_effect_scope_id=_SCOPE_1,
            side_effect_scope_digest=_SCOPE_DIGEST_1,
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )


def _eval_gate(
    *,
    action: PolicyAction,
    grant: GovernedContinuationApprovalGrant | None = None,
    port: ExecutionContinuationPort | None = None,
    decision: PolicyDecision | None = None,
) -> tuple[MseHitlEffectGateDisposition, MutableRuntimePolicyEvaluator, Task]:
    evaluator = MutableRuntimePolicyEvaluator(decision or _decision(action=action))
    boundary, membership = _seed_boundary(evaluator)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    if grant is not None:
        task.runtime.governance.governed_continuation_grant = grant
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
        outcome = evaluate_mse_hitl_effect_gate(
            authorization,
            enforcement_request=request,
            task=task,
            continuation_port=port,
        )
    return outcome.disposition, evaluator, task


def test_scenario_a_initial_require_human() -> None:
    disposition, evaluator, _ = _eval_gate(action=PolicyAction.REQUIRE_HUMAN)
    assert disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert len(evaluator.calls) == 1


def test_scenario_b_allow_grant_resumed_proceeds() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, task = _eval_gate(
        action=PolicyAction.ALLOW,
        grant=_grant(),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    assert task.runtime.governance.governed_continuation_grant is None


def test_scenario_d_require_human_again_never_proceeds() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, task = _eval_gate(
        action=PolicyAction.REQUIRE_HUMAN,
        grant=_grant(),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert task.runtime.governance.governed_continuation_grant is None


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (PolicyAction.DENY, MseHitlEffectGateDisposition.BLOCK),
        (PolicyAction.ESCALATE, MseHitlEffectGateDisposition.REQUIRE_HITL),
    ],
)
def test_scenarios_c_e_deny_escalate(
    action: PolicyAction,
    expected: MseHitlEffectGateDisposition,
) -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, _ = _eval_gate(action=action, grant=_grant(), port=port)
    assert disposition is expected


def test_scenario_f_grant_without_continuation_blocks() -> None:
    disposition, _, _ = _eval_gate(
        action=PolicyAction.ALLOW,
        grant=_grant(),
        port=_FakeContinuationPort(),
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


@pytest.mark.parametrize(
    "state",
    [
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.CANCELLED,
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.ESCALATED,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
    ],
)
def test_wrong_lifecycle_blocks(state: ExecutionContinuationLifecycleState) -> None:
    port = _FakeContinuationPort()
    port.seed(state=state)
    disposition, _, _ = _eval_gate(action=PolicyAction.ALLOW, grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_scenario_j_resume_authorized_without_grant_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUME_AUTHORIZED)
    disposition, _, _ = _eval_gate(action=PolicyAction.ALLOW, grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_scenario_k_wrong_continuation_id_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(continuation_id="gcr_other", state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, _ = _eval_gate(action=PolicyAction.ALLOW, grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", mint_task_id()),
        ("run_id", mint_run_id()),
        ("attempt_id", mint_attempt_id()),
        ("execution_id", mint_execution_id()),
    ],
)
def test_scenario_l_lineage_mismatch_blocks(field: str, value: str) -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, _ = _eval_gate(
        action=PolicyAction.ALLOW,
        grant=_grant(**{field: value}),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_scenario_n_bundle_mismatch_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, _ = _eval_gate(
        action=PolicyAction.ALLOW,
        grant=_grant(policy_bundle_digest=_BUNDLE_D2),
        port=port,
        decision=_decision(action=PolicyAction.ALLOW, policy_bundle_digest=_BUNDLE_D1),
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_freshness_two_evaluations() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    request = _enforcement_request(membership)
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        first = boundary.authorize(request)
        assert first.decision.action is PolicyAction.REQUIRE_HUMAN
        evaluator.set_decision(_decision(action=PolicyAction.ALLOW))
        task.runtime.governance.governed_continuation_grant = _grant()
        second = boundary.authorize(request)
        gate = evaluate_mse_hitl_effect_gate(
            second,
            enforcement_request=request,
            task=task,
            continuation_port=port,
        )
    assert gate.disposition is MseHitlEffectGateDisposition.PROCEED
    assert len(evaluator.calls) == 2


async def test_topology_continue_full_authority_once() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ALLOW))
    boundary, membership = _seed_boundary(evaluator)
    effects = [0]
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)

    class _Inner:
        async def continue_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "done"

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    executor = GovernedOrchestrationSlotContinuationExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _s, _p: _enforcement_request(membership),
        continuation_port=port,
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(task)
    try:
        with bound_gr3_active_execution(
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        ):
            result = await executor.continue_slot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "done"
    assert effects[0] == 1


async def test_topology_continue_require_human_zero_calls() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    effects = [0]
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)

    class _Inner:
        async def continue_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "done"

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    executor = GovernedOrchestrationSlotContinuationExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _s, _p: _enforcement_request(membership),
        continuation_port=port,
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(task)
    try:
        with bound_gr3_active_execution(
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        ):
            with pytest.raises(Exception):
                await executor.continue_slot(
                    slot_id=OrchestrationSlotId("slot-a"),
                    payload=object(),
                )
    finally:
        governed.reset(token)
    assert effects[0] == 0


def test_custom_continuation_port_replaceable() -> None:
    port: ExecutionContinuationPort = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _, _ = _eval_gate(action=PolicyAction.ALLOW, grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED
