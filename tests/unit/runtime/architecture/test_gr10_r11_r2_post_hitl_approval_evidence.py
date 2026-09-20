# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R2 — post-HITL approval evidence vs ordinary ALLOW separation."""

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
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotExecutionError,
    OrchestrationSlotId,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotContinuationExecutor,
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.mse_hitl_effect_gate import (
    EffectContinuationClassification,
    MseHitlEffectGateDisposition,
    evaluate_mse_hitl_effect_gate,
    resolve_effect_continuation_context,
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

_TENANT = "tenant-r11r2"
_WORKSPACE = "workspace-r11r2"
_ACTING = "principal-r11r2"
_OPERATION = "orch.hitl.effect.r2"
_OPERATION_B = "orch.hitl.effect.r2.slot-b"
_RESOURCE = "resource-r11r2"
_NOW = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_ATTEMPT_ID = mint_attempt_id()
_EXECUTION_ID = mint_execution_id()
_POLICY_RULE = "runtime.hitl.r11r2"
_BUNDLE_ID = "bundle-r11r2"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_D1 = "sha256:" + ("33" * 32)
_BUNDLE_D2 = "sha256:" + ("44" * 32)
_SCOPE_1 = "side-effect-scope-r11r2-a"
_SCOPE_2 = "side-effect-scope-r11r2-b"
_SCOPE_DIGEST_1 = "sha256:" + ("cd" * 32)
_SCOPE_DIGEST_2 = "sha256:" + ("ef" * 32)
_CONTINUATION_ID = "gcr_r11r2_test"


def _decision(
    *,
    action: PolicyAction = PolicyAction.ALLOW,
    policy_bundle_digest: str = _BUNDLE_D1,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="r11r2-test",
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=policy_bundle_digest,
    )


def _correlation(
    *,
    continuation_id: str = _CONTINUATION_ID,
    operation_id: str = _OPERATION,
    side_effect_scope_id: str = _SCOPE_1,
    side_effect_scope_digest: str = _SCOPE_DIGEST_1,
) -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.COMPLIANCE,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        operation_id=operation_id,
        resource_scope=_RESOURCE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
    )


def _grant(**overrides: object) -> GovernedContinuationApprovalGrant:
    payload = {
        "grant_id": "gcg_r11r2_test",
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
        "pause_id": "pause-r11r2",
        "human_request_id": "hr-r11r2",
        "approved_at": "2026-09-20T00:00:00+00:00",
    }
    payload.update(overrides)
    return GovernedContinuationApprovalGrant.model_validate(payload)


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
        human_request_id: str | None = "hr-r11r2",
        reason: ContinuationReason = ContinuationReason.COMPLIANCE,
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
            reason=reason,
            governed_correlation=correlation,
            pause_id="pause-r11r2",
            human_request_id=human_request_id,
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
                item
                for item in self._store.values()
                if lookup.identity is not None
                and item.identity.task_id == lookup.identity.task_id
                and item.identity.run_id == lookup.identity.run_id
                and item.identity.attempt_id == lookup.identity.attempt_id
                and item.identity.execution_id == lookup.identity.execution_id
            ]
            if not matches:
                raise ExecutionContinuationError(
                    "continuation not found",
                    code=ExecutionContinuationErrorCode.NOT_FOUND,
                )
            pending = matches[0]
        if lookup.identity is not None:
            assert_execution_continuation_identity_match(
                pending.identity,
                lookup.identity,
                label="lookup",
            )
        return pending

    def advance_to_paused(
        self, lookup: ExecutionContinuationLookup, *, expected_revision: int
    ) -> PendingExecutionContinuation:
        raise ExecutionContinuationError(
            "not used",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )

    def advance_to_human_wait(
        self, lookup: ExecutionContinuationLookup, *, expected_revision: int
    ) -> PendingExecutionContinuation:
        raise ExecutionContinuationError(
            "not used",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )

    def apply_resolution(
        self, command: ExecutionContinuationResolutionCommand
    ) -> PendingExecutionContinuation:
        pending = self.get_pending(
            ExecutionContinuationLookup(
                continuation_id=command.continuation_id,
                identity=command.identity,
            )
        )
        updated = apply_resolution_to_pending(pending, command)
        self._store[updated.continuation_id] = updated
        return updated

    def resume(
        self, command: ExecutionContinuationResumeCommand
    ) -> PendingExecutionContinuation:
        pending = self.get_pending(
            ExecutionContinuationLookup(
                continuation_id=command.continuation_id,
                identity=command.identity,
            )
        )
        updated = apply_resume_to_pending(pending, command)
        self._store[updated.continuation_id] = updated
        return updated

    def cancel_continuation(
        self, lookup: ExecutionContinuationLookup, *, expected_revision: int
    ) -> PendingExecutionContinuation:
        raise ExecutionContinuationError(
            "not used",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )


def _seed_boundary(
    evaluator: MutableRuntimePolicyEvaluator,
    *,
    operation_id: str = _OPERATION,
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, WorkspaceMembership]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-r11r2",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-r11r2",
            principal_id=_ACTING,
            authority_scopes=("document.delete",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    for rule_id, layer, resource in (
        ("workspace-allow-r11r2", PolicyCompositionLayer.WORKSPACE_POLICY, None),
        ("resource-allow-r11r2", PolicyCompositionLayer.RESOURCE_POLICY, _RESOURCE),
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
            operation_id=operation_id,
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
    operation_id: str = _OPERATION,
    scope_id: str = _SCOPE_1,
    scope_digest: str = _SCOPE_DIGEST_1,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=operation_id,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=operation_id,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id=scope_id,
            side_effect_scope_digest=scope_digest,
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
    action: PolicyAction = PolicyAction.ALLOW,
    grant: GovernedContinuationApprovalGrant | None = None,
    port: ExecutionContinuationPort | None = None,
    decision: PolicyDecision | None = None,
    operation_id: str = _OPERATION,
    scope_id: str = _SCOPE_1,
    scope_digest: str = _SCOPE_DIGEST_1,
) -> tuple[MseHitlEffectGateDisposition, Task]:
    evaluator = MutableRuntimePolicyEvaluator(decision or _decision(action=action))
    boundary, membership = _seed_boundary(evaluator, operation_id=operation_id)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    if grant is not None:
        task.runtime.governance.governed_continuation_grant = grant
    request = _enforcement_request(
        membership,
        operation_id=operation_id,
        scope_id=scope_id,
        scope_digest=scope_digest,
    )
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
    return outcome.disposition, task


# --- Scenario A: ordinary ALLOW ---


def test_scenario_a_ordinary_allow_no_continuation_no_grant_proceeds() -> None:
    disposition, _ = _eval_gate(port=_FakeContinuationPort())
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_scenario_a_ordinary_allow_without_port_proceeds() -> None:
    disposition, _ = _eval_gate(port=None)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


# --- Scenario B: post-HITL happy path ---


def test_scenario_b_post_hitl_resumed_matching_grant_proceeds() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, task = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    assert task.runtime.governance.governed_continuation_grant is None


# --- Scenario C: missing grant ---


def test_scenario_c_post_hitl_resumed_missing_grant_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK


# --- Scenario D: wrong grant ---


def test_scenario_d_post_hitl_wrong_grant_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(
        grant=_grant(side_effect_scope_id=_SCOPE_2, side_effect_scope_digest=_SCOPE_DIGEST_2),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


# --- Scenario E: wrong bundle ---


def test_scenario_e_post_hitl_wrong_bundle_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(
        grant=_grant(policy_bundle_digest=_BUNDLE_D2),
        port=port,
        decision=_decision(policy_bundle_digest=_BUNDLE_D1),
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


# --- Scenarios F–I: wrong lifecycle ---


@pytest.mark.parametrize(
    "state",
    [
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.CANCELLED,
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.ESCALATED,
    ],
)
def test_scenarios_f_i_post_hitl_non_resumed_blocks(
    state: ExecutionContinuationLifecycleState,
) -> None:
    port = _FakeContinuationPort()
    port.seed(state=state)
    disposition, _ = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK
    disposition_no_grant, _ = _eval_gate(grant=None, port=port)
    assert disposition_no_grant is MseHitlEffectGateDisposition.BLOCK


# --- Scenario J: non-HITL continuation ---


def test_scenario_j_resumed_non_hitl_continuation_no_grant_proceeds() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=None,
        human_request_id=None,
        reason=ContinuationReason.QUOTE,
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=MeaningfulSideEffectRequest(
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
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert (
        context.classification
        is EffectContinuationClassification.CANONICAL_NON_HITL_OR_NON_BLOCKING
    )
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


# --- Scenario K: stale unrelated grant ---


def test_scenario_k_stale_unrelated_grant_does_not_block_ordinary_allow() -> None:
    disposition, task = _eval_gate(
        grant=_grant(
            side_effect_scope_id="other-scope",
            side_effect_scope_digest=_SCOPE_DIGEST_2,
            continuation_request_id="gcr_unrelated",
        ),
        port=_FakeContinuationPort(),
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    assert task.runtime.governance.governed_continuation_grant is not None


# --- Scenario L: cross-slot ---


def test_scenario_l_cross_slot_hitl_does_not_govern_other_operation() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(operation_id=_OPERATION),
    )
    # Slot B ordinary ALLOW — different operation; HITL continuation is for slot A.
    disposition, _ = _eval_gate(
        operation_id=_OPERATION_B,
        grant=None,
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


# --- Scenario M: new effect same execution, different scope ---


def test_scenario_m_new_scope_same_execution_not_auto_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            side_effect_scope_id=_SCOPE_1,
            side_effect_scope_digest=_SCOPE_DIGEST_1,
        ),
    )
    disposition, _ = _eval_gate(
        grant=None,
        port=port,
        scope_id=_SCOPE_2,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_scenario_m_old_grant_cannot_authorize_new_scope() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            side_effect_scope_id=_SCOPE_2,
            side_effect_scope_digest=_SCOPE_DIGEST_2,
            continuation_id="gcr_scope2",
        ),
        continuation_id="gcr_scope2",
    )
    disposition, _ = _eval_gate(
        grant=_grant(),  # scoped to SCOPE_1
        port=port,
        scope_id=_SCOPE_2,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


# --- Classification helper ---


def test_classification_no_continuation() -> None:
    context = resolve_effect_continuation_context(
        _FakeContinuationPort(),
        side_effect=MeaningfulSideEffectRequest(
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
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert (
        context.classification is EffectContinuationClassification.NO_CANONICAL_CONTINUATION
    )


def test_classification_post_hitl_resumed() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    context = resolve_effect_continuation_context(
        port,
        side_effect=MeaningfulSideEffectRequest(
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
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.POST_HITL_RESUMED


# --- Topology ---


async def test_topology_ordinary_allow_executes_once() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    effects = [0]

    class _Inner:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "done"

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _s, _p: _enforcement_request(membership),
        continuation_port=_FakeContinuationPort(),
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(task)
    try:
        with bound_gr3_active_execution(
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        ):
            result = await executor.execute_slot(
                slot_id=OrchestrationSlotId("slot-ordinary"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "done"
    assert effects[0] == 1


async def test_topology_post_hitl_continue_with_grant_once() -> None:
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
            result = await executor.continue_slot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "done"
    assert effects[0] == 1
    assert task.runtime.governance.governed_continuation_grant is None


async def test_topology_post_hitl_continue_missing_grant_zero_effect() -> None:
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
            with pytest.raises(OrchestrationSlotExecutionError):
                await executor.continue_slot(
                    slot_id=OrchestrationSlotId("slot-a"),
                    payload=object(),
                )
    finally:
        governed.reset(token)
    assert effects[0] == 0


# --- Pluginability ---


def test_custom_continuation_port_replaceable() -> None:
    port: ExecutionContinuationPort = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


# --- DENY / REQUIRE_HUMAN / ESCALATE regression ---


def test_require_human_blocks_despite_resumed_grant() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, task = _eval_gate(
        action=PolicyAction.REQUIRE_HUMAN,
        grant=_grant(),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
    assert task.runtime.governance.governed_continuation_grant is None


def test_deny_blocks_and_does_not_consume_unrelated() -> None:
    disposition, _ = _eval_gate(action=PolicyAction.DENY, grant=None, port=_FakeContinuationPort())
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_escalate_require_hitl() -> None:
    disposition, _ = _eval_gate(
        action=PolicyAction.ESCALATE,
        grant=None,
        port=_FakeContinuationPort(),
    )
    assert disposition is MseHitlEffectGateDisposition.REQUIRE_HITL
