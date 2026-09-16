# © Artur Czarnecki. All rights reserved.

"""GR-6-RS1 — Decision subject ↔ side-effect resource binding at canonical boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
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
from intergrax.contracts.canonical_inner_governance import CanonicalInnerGovernanceViolation
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
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.decision_governance_material import (
    assert_decision_governance_material_bound,
)
from intergrax.runtime.execution.decision_governed_side_effect import (
    authorize_and_execute_decision_bound_side_effect,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
    decision_governed_side_effect_requirement_policy,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-gr6rs1"
_WORKSPACE = "workspace-gr6rs1"
_ACTING = "principal-gr6rs1"
_SCOPE = "gr6rs1.scope"
_RESOURCE_A = "resource-gr6rs1-a"
_RESOURCE_B = "resource-gr6rs1-b"
_PAYMENTS_CAPTURE = "payments.capture"
_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Payload:
    value: str


def _accepted(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    action_kind: str,
    subject: str,
) -> tuple[AuthoritativeAcceptedDecision[_Payload], object, object]:
    version = initial_decision_version()
    decision = AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="gr6rs1", subject=subject),
            tenant_id=_TENANT,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr6rs1.payload"),
            content=_Payload(value="ok"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(version)),
    )
    action = decision_execution_action(kind=action_kind, subject=subject)
    policy = decision_governance_policy_context(policy_provenance_digest="gr6rs1-policy")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy,
        tenant_id=_TENANT,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    return decision, authorization, action


def _material_for(
    decision: AuthoritativeAcceptedDecision[_Payload],
    action: object,
) -> object:
    return decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )


def _side_effect(
    *,
    action: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    resource: str | None,
    material: object | None = None,
) -> MeaningfulSideEffectRequest:
    base = MeaningfulSideEffectRequest(
        action=action,
        kinds=(MeaningfulSideEffectKind.COMMITMENT,),
        side_effect_scope_id="scope-gr6rs1",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_TENANT,
        principal_id=_ACTING,
        resource=resource,
    )
    if material is None:
        return base
    return base.model_copy(update={"decision_governance_material": material})


def _gate(*, operation_id: str) -> CollaborativeWorkEnforcementGate:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-gr6rs1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-gr6rs1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-gr6rs1",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        ),
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=operation_id,
            authority_scope=_SCOPE,
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
                    rule_id="runtime.gr6rs1",
                    action=operation_id,
                    decision=PolicyAction.ALLOW,
                ),
            ),
        ),
    )


def _boundary(
    *,
    task_id: TaskId,
    operation_id: str,
    required_actions: frozenset[str],
) -> MeaningfulSideEffectAuthorizationBoundary:
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(operation_id=operation_id),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=decision_governed_side_effect_requirement_policy(
            required_actions=required_actions,
        ),
    )


def _enforcement(
    side_effect: MeaningfulSideEffectRequest,
    *,
    operation_id: str,
    resource_scope: str | None,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=operation_id,
        acting_principal_id=_ACTING,
        resource_scope=resource_scope,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


def test_subject_resource_match_passes_binding() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision, _, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=ACTION_ACCEPT_QUOTE,
        subject=_RESOURCE_A,
    )
    material = _material_for(decision, action)
    request = _side_effect(
        action=ACTION_ACCEPT_QUOTE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
        material=material,
    )
    assert_decision_governance_material_bound(request)


def test_wrong_resource_denied_at_boundary_no_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision, authorization, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=ACTION_ACCEPT_QUOTE,
        subject=_RESOURCE_A,
    )
    material = _material_for(decision, action)
    boundary = _boundary(
        task_id=task_id,
        operation_id=ACTION_ACCEPT_QUOTE,
        required_actions=frozenset({ACTION_ACCEPT_QUOTE}),
    )
    side_effect = _side_effect(
        action=ACTION_ACCEPT_QUOTE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_B,
        material=material,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            _enforcement(
                side_effect,
                operation_id=ACTION_ACCEPT_QUOTE,
                resource_scope=_RESOURCE_B,
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert "bound_action_subject" in (result.decision.reason or "")
    assert calls == []


def test_same_lineage_wrong_resource_only_denied() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision, authorization, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=ACTION_ACCEPT_QUOTE,
        subject=_RESOURCE_A,
    )
    material = _material_for(decision, action)
    side_effect = _side_effect(
        action=ACTION_ACCEPT_QUOTE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_B,
        material=material,
    )
    with pytest.raises(CanonicalInnerGovernanceViolation, match="bound_action_subject"):
        assert_decision_governance_material_bound(side_effect)


def test_missing_resource_with_material_denied() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision, _, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=ACTION_ACCEPT_QUOTE,
        subject=_RESOURCE_A,
    )
    material = _material_for(decision, action)
    side_effect = _side_effect(
        action=ACTION_ACCEPT_QUOTE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=None,
        material=material,
    )
    with pytest.raises(CanonicalInnerGovernanceViolation, match="resource identity"):
        assert_decision_governance_material_bound(side_effect)


def test_valid_decision_bound_allow_executes_once() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision, authorization, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=ACTION_ACCEPT_QUOTE,
        subject=_RESOURCE_A,
    )
    policy = decision_governance_policy_context(policy_provenance_digest="gr6rs1-policy")
    boundary = _boundary(
        task_id=task_id,
        operation_id=ACTION_ACCEPT_QUOTE,
        required_actions=frozenset({ACTION_ACCEPT_QUOTE}),
    )
    side_effect = _side_effect(
        action=ACTION_ACCEPT_QUOTE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=_RESOURCE_A,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        outcome = authorize_and_execute_decision_bound_side_effect(
            boundary,
            enforcement_request=_enforcement(
                side_effect,
                operation_id=ACTION_ACCEPT_QUOTE,
                resource_scope=_RESOURCE_A,
            ),
            decision=decision,
            authorization=authorization,
            action=action,
            policy_context=policy,
            execute=lambda: calls.append("effect") or "ok",
        )
    assert outcome == "ok"
    assert calls == ["effect"]


def test_not_required_create_without_material_unaffected() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(operation_id=ACTION_CREATE_EXTERNAL_WORK),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
    )
    side_effect = _side_effect(
        action=ACTION_CREATE_EXTERNAL_WORK,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=None,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        outcome = boundary.authorize_and_execute(
            _enforcement(
                side_effect,
                operation_id=ACTION_CREATE_EXTERNAL_WORK,
                resource_scope=None,
            ),
            lambda: calls.append("create") or "created",
        )
    assert outcome == "created"
    assert calls == ["create"]


def test_payments_capture_neutral_subject_resource_binding() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    payment_resource = "payment-resource-1"
    decision, _, action = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        action_kind=_PAYMENTS_CAPTURE,
        subject=payment_resource,
    )
    material = _material_for(decision, action)
    matching = _side_effect(
        action=_PAYMENTS_CAPTURE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        resource=payment_resource,
        material=material,
    )
    assert_decision_governance_material_bound(matching)
    wrong = matching.model_copy(update={"resource": "payment-resource-2"})
    with pytest.raises(CanonicalInnerGovernanceViolation, match="bound_action_subject"):
        assert_decision_governance_material_bound(wrong)
