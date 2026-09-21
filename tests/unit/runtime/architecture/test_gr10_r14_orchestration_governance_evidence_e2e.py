# © Artur Czarnecki. All rights reserved.

"""GR-10-R14 — orchestration Governance Evidence enterprise proofs."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
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
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.governance.governance_evidence_composition import (
    build_in_memory_governance_evidence_persistence,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.governance.orchestration_governance_evidence_composition import (
    OrchestrationGovernanceEvidenceCompositionError,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from tests.qualification.multiplayer.mp7b.composition import compose_allow_platform_port
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)
from tests.unit.runtime.policy.test_g5c2b2b_governed_side_effect_reauthorization import (
    MutableRuntimePolicyEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-r14"
_TENANT_B = "tenant-r14-b"
_WORKSPACE = "workspace-r14"
_ACTING = "principal-r14"
_OPERATION = "mp7b.proof.mutation"
_SCOPE = "mp7b.proof.scope"
_RESOURCE = "mp7b-resource-1"


class _CapturingPersistence(GovernanceEvidencePersistencePort):
    def __init__(self) -> None:
        self.facts: list[GovernanceDecisionEvidenceFact] = []
        self.fail = False

    def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
        if self.fail:
            return GovernanceEvidencePersistenceOutcome(
                persisted=False,
                evidence_id=fact.evidence_id,
                error_code="test_failure",
            )
        self.facts.append(fact)
        return GovernanceEvidencePersistenceOutcome(persisted=True, evidence_id=fact.evidence_id)


def _seeded_boundary(
    store: _CapturingPersistence,
    *,
    tenant_id: str = _TENANT,
    runtime_policy_evaluator: MutableRuntimePolicyEvaluator | RuntimePolicyEngine | None = None,
):
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            membership_id=f"r14-membership-{tenant_id}",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            authority_grant_id=f"r14-grant-{tenant_id}",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            policy_rule_id=f"r14-ws-allow-{tenant_id}",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            policy_rule_id=f"r14-res-allow-{tenant_id}",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.REQUIRED,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    evaluator = (
        runtime_policy_evaluator
        if runtime_policy_evaluator is not None
        else RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="r14.runtime.allow",
                    action=_OPERATION,
                    decision=PolicyAction.ALLOW,
                ),
            ),
        )
    )
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        profile_repository=profile_repo,
        membership_repository=membership_repo,
        principal_authority_repository=authority_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        collaborative_policy_repository=policy_repo,
        runtime_policy_evaluator=evaluator,
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        governance_evidence_persistence=store,
        production_mode=True,
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="r14-side-effect-scope",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            principal_id=_ACTING,
            tenant_id=tenant_id,
            resource=_RESOURCE,
        ),
    )
    return boundary, request, run_id, attempt_id, execution_id


def test_mse_allow_emits_typed_fact_with_execution_correlation() -> None:
    store = _CapturingPersistence()
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(store)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize(request)
    assert result.decision.action is PolicyAction.ALLOW
    assert len(store.facts) == 1
    fact = store.facts[0]
    assert fact.evaluation_point is GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT
    assert fact.decision is PolicyAction.ALLOW
    assert fact.has_full_execution_correlation
    assert fact.tenant_id == _TENANT


def test_mse_deny_emits_fact_without_permitting_effect() -> None:
    store = _CapturingPersistence()
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(
        store,
        runtime_policy_evaluator=MutableRuntimePolicyEvaluator(
            PolicyDecision(action=PolicyAction.DENY, reason="deny"),
        ),
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize(request)
    assert result.permitted is False
    assert store.facts[0].decision is PolicyAction.DENY


def test_persistence_failure_does_not_flip_allow() -> None:
    store = _CapturingPersistence()
    store.fail = True
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(store)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize(request)
    assert result.decision.action is PolicyAction.ALLOW


def test_production_boundary_requires_governance_evidence_persistence() -> None:
    with pytest.raises(OrchestrationGovernanceEvidenceCompositionError):
        build_production_orchestration_meaningful_side_effect_authorization_boundary(
            profile_repository=InMemoryCollaborativeOperationPolicyProfileRepository(),
            membership_repository=InMemoryWorkspaceMembershipRepository(),
            principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            collaborative_policy_repository=InMemoryCollaborativePolicyRepository(),
            runtime_policy_evaluator=RuntimePolicyEngine(),
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
            production_mode=True,
        )


def test_custom_persistence_port_without_runtime_modification() -> None:
    store = build_in_memory_governance_evidence_persistence()
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(store)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        boundary.authorize(request)
    assert len(store.facts) == 1


def test_same_evaluation_replay_preserves_evidence_id() -> None:
    store = _CapturingPersistence()
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(store)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        boundary.authorize(request)
        boundary.authorize(request)
    assert len(store.facts) == 2
    assert store.facts[0].evidence_id == store.facts[1].evidence_id


def test_cross_tenant_produces_distinct_evidence_id() -> None:
    store_a = _CapturingPersistence()
    store_b = _CapturingPersistence()
    boundary_a, request_a, run_id_a, attempt_id_a, execution_id_a = _seeded_boundary(store_a)
    boundary_b, request_b, run_id_b, attempt_id_b, execution_id_b = _seeded_boundary(
        store_b,
        tenant_id=_TENANT_B,
    )
    with bound_gr3_active_execution(
        run_id=run_id_a,
        attempt_id=attempt_id_a,
        execution_id=execution_id_a,
    ):
        boundary_a.authorize(request_a)
    with bound_gr3_active_execution(
        run_id=run_id_b,
        attempt_id=attempt_id_b,
        execution_id=execution_id_b,
    ):
        boundary_b.authorize(request_b)
    assert store_a.facts[0].evidence_id != store_b.facts[0].evidence_id


def test_escalate_emits_canonical_fact() -> None:
    store = _CapturingPersistence()
    boundary, request, run_id, attempt_id, execution_id = _seeded_boundary(
        store,
        runtime_policy_evaluator=MutableRuntimePolicyEvaluator(
            PolicyDecision(action=PolicyAction.ESCALATE, reason="escalate"),
        ),
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize(request)
    assert result.decision.action is PolicyAction.ESCALATE
    assert store.facts[0].decision is PolicyAction.ESCALATE


def test_mp7b_compose_allow_still_qualified_path() -> None:
    port, request = compose_allow_platform_port()
    side_effect = request.meaningful_side_effect_request
    assert side_effect is not None
    with bound_gr3_active_execution(
        run_id=side_effect.run_id,
        attempt_id=side_effect.attempt_id,
        execution_id=side_effect.execution_id,
    ):
        result = port.authorize(request)
    assert result.decision.action is PolicyAction.ALLOW
