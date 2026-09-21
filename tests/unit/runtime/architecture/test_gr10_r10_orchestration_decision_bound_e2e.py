# © Artur Czarnecki. All rights reserved.

"""GR-10-R10 — orchestration decision-bound consequential effect E2E proofs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_execution_authorization,
    decision_governance_policy_context,
)
from intergrax.contracts.decision_governance_material import (
    decision_governance_material_ref_from_accepted,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.decision_governed_side_effect import (
    attach_decision_governance_material,
    authorize_and_execute_decision_bound_side_effect,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    ConfiguredDecisionRequirementPolicy,
    DecisionRequirementRule,
    PermissiveDecisionRequirementPolicy,
    decision_governed_side_effect_requirement_policy,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    OrchestrationDecisionBoundCompositionError,
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.nexus.orchestration.orchestration_graph_meaningful_side_effect import (
    build_orchestration_graph_slot_enforcement_request,
    build_orchestration_graph_slot_meaningful_side_effect_request,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_host_task,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from testing_support.orchestration_governance_evidence_wiring import (
    default_test_orchestration_evidence_persistence,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-gr10r10"
_WORKSPACE = "workspace-gr10r10"
_ACTING = "principal-gr10r10"
_OPERATION = "orch.slot.effect"
_SCOPE_A = "orchestration/topology/slot/slot-a"
_SCOPE_B = "orchestration/topology/slot/slot-b"
_RESOURCE_A = "resource-slot-a"
_RESOURCE_B = "resource-slot-b"
_NOW = datetime(2026, 9, 19, 10, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Payload:
    value: str


def _gate(runtime_decision: PolicyAction) -> CollaborativeWorkEnforcementGate:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-gr10r10",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-gr10r10",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE_A, _SCOPE_B),
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-gr10r10",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE_A,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        ),
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            authority_scope=_SCOPE_A,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        ),
    )
    return CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.gr10r10",
                    action=_OPERATION,
                    decision=runtime_decision,
                ),
            ),
        ),
    )


def _side_effect(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    scope_id: str,
    resource: str,
    material: object | None = None,
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=scope_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id=_ACTING,
        tenant_id=_TENANT,
        resource=resource,
        decision_governance_material=material,
    )


def _enforcement(side_effect: MeaningfulSideEffectRequest) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=side_effect.resource,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


def _accepted(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    resource: str,
) -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    return AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="gr10r10", subject=resource),
            tenant_id=_TENANT,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr10r10.payload"),
            content=_Payload(value="ok"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(version)),
    )


def _orchestration_boundary(
    *,
    task_id: TaskId,
    runtime_decision: PolicyAction,
    decision_requirement_policy: DecisionRequirementPolicy,
) -> MeaningfulSideEffectAuthorizationBoundary:
    return build_production_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=InMemoryCollaborativeOperationPolicyProfileRepository(),
        membership_repository=InMemoryWorkspaceMembershipRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        collaborative_policy_repository=InMemoryCollaborativePolicyRepository(),
        runtime_policy_evaluator=RuntimePolicyEngine(),
        decision_requirement_policy=decision_requirement_policy,
        inner_execution_guard=default_gr3_inner_guard(task_id),
        production_mode=True,
        governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
    )


def test_production_orchestration_composition_binds_explicit_policy() -> None:
    policy = PermissiveDecisionRequirementPolicy()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = _orchestration_boundary(
        task_id=task_id,
        runtime_decision=PolicyAction.ALLOW,
        decision_requirement_policy=policy,
    )
    assert boundary._decision_requirement_policy is policy


def test_missing_production_orchestration_decision_policy_fails_closed() -> None:
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        build_production_orchestration_meaningful_side_effect_authorization_boundary(
            profile_repository=InMemoryCollaborativeOperationPolicyProfileRepository(),
            membership_repository=InMemoryWorkspaceMembershipRepository(),
            principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            collaborative_policy_repository=InMemoryCollaborativePolicyRepository(),
            runtime_policy_evaluator=RuntimePolicyEngine(),
            decision_requirement_policy=None,
            production_mode=True,
            governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
        )


def test_not_required_policy_allows_effect_without_decision_material() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=_SCOPE_A,
                    resource=_RESOURCE_A,
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert result == "ok"
    assert calls == ["effect"]


def test_mse_allow_without_required_decision_zero_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=decision_governed_side_effect_requirement_policy(
            required_actions=frozenset({_OPERATION}),
        ),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=_SCOPE_A,
                    resource=_RESOURCE_A,
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert "decision provenance required but absent" in result.decision.reason
    assert calls == []


def test_required_valid_decision_mse_allow_one_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
    )
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE_A)
    policy_ctx = decision_governance_policy_context(policy_provenance_digest="gr10r10")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy_ctx,
        tenant_id=_TENANT,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=decision_governed_side_effect_requirement_policy(
            required_actions=frozenset({_OPERATION}),
        ),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = authorize_and_execute_decision_bound_side_effect(
            boundary,
            enforcement_request=_enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=_SCOPE_A,
                    resource=_RESOURCE_A,
                ),
            ),
            decision=decision,
            authorization=authorization,
            action=action,
            policy_context=policy_ctx,
            execute=lambda: calls.append("effect") or "ok",
        )
    assert result == "ok"
    assert calls == ["effect"]


def test_valid_decision_mse_deny_zero_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
    )
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE_A)
    policy_ctx = decision_governance_policy_context(policy_provenance_digest="gr10r10")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy_ctx,
        tenant_id=_TENANT,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.DENY),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=decision_governed_side_effect_requirement_policy(
            required_actions=frozenset({_OPERATION}),
        ),
    )
    material = decision_governance_material_ref_from_accepted(decision=decision, action=action)
    side_effect = attach_decision_governance_material(
        _side_effect(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            scope_id=_SCOPE_A,
            resource=_RESOURCE_A,
            material=material,
        ),
        decision=decision,
        action=action,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = authorize_and_execute_decision_bound_side_effect(
            boundary,
            enforcement_request=_enforcement(side_effect),
            decision=decision,
            authorization=authorization,
            action=action,
            policy_context=policy_ctx,
            execute=lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_wrong_scope_decision_zero_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
    )
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE_A)
    material = decision_governance_material_ref_from_accepted(decision=decision, action=action)
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=ConfiguredDecisionRequirementPolicy(
            rules=(
                DecisionRequirementRule(
                    side_effect_scope_id=f"{_SCOPE_B}:effect",
                    action=_OPERATION,
                ),
            ),
        ),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        allowed = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=f"{_SCOPE_A}:effect",
                    resource=_RESOURCE_A,
                ),
            ),
            lambda: calls.append("a") or "a",
        )
        blocked = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=f"{_SCOPE_B}:effect",
                    resource=_RESOURCE_B,
                    material=material,
                ),
            ),
            lambda: calls.append("b") or "b",
        )
    assert allowed == "a"
    assert isinstance(blocked, MeaningfulSideEffectAuthorizationResult)
    assert blocked.decision.action is PolicyAction.DENY
    assert calls == ["a"]


def test_cross_slot_decision_reuse_blocked() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
    )
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE_A)
    material = decision_governance_material_ref_from_accepted(decision=decision, action=action)
    policy = ConfiguredDecisionRequirementPolicy(
        rules=(
            DecisionRequirementRule(
                side_effect_scope_id=f"{_SCOPE_A}:effect",
                action=_OPERATION,
            ),
            DecisionRequirementRule(
                side_effect_scope_id=f"{_SCOPE_B}:effect",
                action=_OPERATION,
            ),
        ),
    )
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=policy,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        first = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=f"{_SCOPE_A}:effect",
                    resource=_RESOURCE_A,
                    material=material,
                ),
            ),
            lambda: calls.append("a") or "a",
        )
        second = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    scope_id=f"{_SCOPE_B}:effect",
                    resource=_RESOURCE_B,
                    material=material,
                ),
            ),
            lambda: calls.append("b") or "b",
        )
    assert first == "a"
    assert isinstance(second, MeaningfulSideEffectAuthorizationResult)
    assert second.decision.action is PolicyAction.DENY
    assert calls == ["a"]


@pytest.mark.asyncio
async def test_topology_slot_authorize_reevaluates_decision_requirement() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_ACTING,
        ),
    )
    host_task = build_orchestration_topology_host_task(
        tenant_id=_TENANT,
        user_id=_ACTING,
        task_id=task_id,
    )
    governed = ActiveGovernedExecutionTask()
    governed_token = governed.bind(host_task)
    policy = decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({_OPERATION}),
    )
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=policy,
    )
    slot_id = OrchestrationSlotId("slot-recovery")
    scope = f"orchestration/topology/slot/{slot_id}"
    side_effect = build_orchestration_graph_slot_meaningful_side_effect_request(
        slot_id=slot_id,
        operation_id=f"slot:{slot_id}",
        resource_scope=scope,
        side_effect_scope_id=f"{scope}:effect",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
    )
    enforcement_request = build_orchestration_graph_slot_enforcement_request(
        slot_id=slot_id,
        side_effect=side_effect,
        operation_id=f"slot:{slot_id}",
        resource_scope=scope,
    )
    effects = 0

    class _InnerSlot:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: _Payload) -> str:
            nonlocal effects
            effects += 1
            return "ok"

    executor = GovernedOrchestrationSlotExecutor(
        inner=_InnerSlot(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _sid, _payload: enforcement_request,
    )
    from intergrax.contracts.orchestration_topology import OrchestrationSlotExecutionError

    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        with pytest.raises(OrchestrationSlotExecutionError):
            await executor.execute_slot(slot_id=slot_id, payload=_Payload(value="x"))
    assert effects == 0
    reset_active_execution_identity(identity_token)
    reset_active_execution_governance_identity(governance_token)
    governed.reset(governed_token)
