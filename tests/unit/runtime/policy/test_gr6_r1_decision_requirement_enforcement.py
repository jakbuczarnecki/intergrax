# © Artur Czarnecki. All rights reserved.

"""GR-6-R1 — Decision requirement enforcement at canonical side-effect boundary."""

from __future__ import annotations

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
from intergrax.contracts.decision_requirement_policy import (
    DecisionRequirement,
    DecisionRequirementContext,
    DecisionRequirementPolicy,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_execution_id,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.decision_governed_side_effect import (
    authorize_and_execute_decision_bound_side_effect,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    ConfiguredDecisionRequirementPolicy,
    DecisionRequirementRule,
    PermissiveDecisionRequirementPolicy,
    decision_governed_side_effect_requirement_policy,
)
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_decision_governed_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT_A = "tenant-gr6r1-a"
_TENANT_B = "tenant-gr6r1-b"
_WORKSPACE = "workspace-gr6r1"
_ACTING = "principal-gr6r1"
_OPERATION = "gr6r1.side_effect"
_SCOPE = "gr6r1.scope"
_RESOURCE = "resource-gr6r1"
_NOW = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Payload:
    value: str


class _FalseyRequiredPolicy:
    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        _ = context
        return DecisionRequirement.REQUIRED


class _ExplodingPolicy:
    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        _ = context
        raise RuntimeError("policy exploded")


def _gate(
    runtime_decision: PolicyAction,
    *,
    tenant_id: str = _TENANT_A,
) -> CollaborativeWorkEnforcementGate:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            membership_id=f"membership-gr6r1-{tenant_id}",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            authority_grant_id=f"grant-gr6r1-{tenant_id}",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            policy_rule_id=f"workspace-allow-gr6r1-{tenant_id}",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        ),
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=tenant_id,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
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
                    rule_id="runtime.gr6r1",
                    action=_OPERATION,
                    decision=runtime_decision,
                ),
            ),
        ),
    )


def _boundary(
    *,
    task_id: TaskId,
    runtime_decision: PolicyAction,
    decision_requirement_policy: DecisionRequirementPolicy | None = None,
) -> MeaningfulSideEffectAuthorizationBoundary:
    policy = decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({_OPERATION}),
    )
    resolved_policy = (
        decision_requirement_policy
        if decision_requirement_policy is not None
        else policy
    )
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(runtime_decision),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=resolved_policy,
    )


def _accepted(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    tenant_id: str = _TENANT_A,
) -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    return AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="gr6r1", subject="effect"),
            tenant_id=tenant_id,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr6r1.payload"),
            content=_Payload(value="ok"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(version)),
    )


def _side_effect(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    tenant_id: str = _TENANT_A,
    material: object | None = None,
) -> MeaningfulSideEffectRequest:
    base = MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope-gr6r1",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=tenant_id,
        principal_id=_ACTING,
        resource=_RESOURCE,
    )
    if material is None:
        return base
    return base.model_copy(update={"decision_governance_material": material})


def _enforcement(
    side_effect: MeaningfulSideEffectRequest,
    *,
    tenant_id: str = _TENANT_A,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


def _authorization_for(decision: AuthoritativeAcceptedDecision[_Payload]):
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE)
    policy = decision_governance_policy_context(policy_provenance_digest="gr6r1-policy")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy,
        tenant_id=decision.identity.tenant_id,
    )
    return decision_execution_authorization(governance_decision=governance), action, policy


def test_direct_boundary_required_without_material_denies_no_effect() -> None:
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
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert "decision provenance required but absent" in result.decision.reason
    assert calls == []


def test_not_required_without_material_preserves_governance_allow() -> None:
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
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert result == "ok"
    assert calls == ["effect"]


def test_coordinator_with_valid_material_and_governance_allow() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision)
    boundary = _boundary(task_id=task_id, runtime_decision=PolicyAction.ALLOW)
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
                ),
            ),
            decision=decision,
            authorization=authorization,
            action=action,
            policy_context=policy,
            execute=lambda: calls.append("effect") or "ok",
        )
    assert result == "ok"
    assert calls == ["effect"]


def test_required_governance_deny_blocks_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision)
    boundary = _boundary(task_id=task_id, runtime_decision=PolicyAction.DENY)
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
                ),
            ),
            decision=decision,
            authorization=authorization,
            action=action,
            policy_context=policy,
            execute=lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_policy_failure_fails_closed() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=_ExplodingPolicy(),
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
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_falsey_custom_policy_still_enforced() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=_FalseyRequiredPolicy(),
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
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_tenant_aware_requirement() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    policy = ConfiguredDecisionRequirementPolicy(
        rules=(DecisionRequirementRule(tenant_id=_TENANT_A, action=_OPERATION),),
    )
    boundary_a = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW, tenant_id=_TENANT_A),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=policy,
    )
    boundary_b = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW, tenant_id=_TENANT_B),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=policy,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        blocked = boundary_a.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    tenant_id=_TENANT_A,
                ),
                tenant_id=_TENANT_A,
            ),
            lambda: calls.append("a") or "a",
        )
        allowed = boundary_b.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    tenant_id=_TENANT_B,
                ),
                tenant_id=_TENANT_B,
            ),
            lambda: calls.append("b") or "b",
        )
    assert isinstance(blocked, MeaningfulSideEffectAuthorizationResult)
    assert blocked.decision.action is PolicyAction.DENY
    assert allowed == "b"
    assert calls == ["b"]


def test_not_required_malformed_material_still_denied() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    action = decision_execution_action(kind="other.op", subject=_RESOURCE)
    bad_material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
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
                    material=bad_material,
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_four_id_mismatch_on_material_denied() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE)
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    other_execution = mint_execution_id()
    boundary = _boundary(task_id=task_id, runtime_decision=PolicyAction.ALLOW)
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
                    execution_id=other_execution,
                    material=material,
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_production_composition_wiring_symbols() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    policy = decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({_OPERATION}),
    )
    boundary = build_decision_governed_meaningful_side_effect_authorization_boundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        decision_requirement_policy=policy,
        task_scope=StaticActiveTaskScope(task_id),
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
                ),
            ),
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_custom_policy_without_subclassing_default() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    custom = ConfiguredDecisionRequirementPolicy(
        rules=(
            DecisionRequirementRule(
                side_effect_scope_id="scope-custom-only",
                action=_OPERATION,
            ),
        ),
    )
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=_gate(PolicyAction.ALLOW),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=custom,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                ).model_copy(update={"side_effect_scope_id": "scope-custom-only"}),
            ),
            lambda: calls.append("x") or "x",
        )
        allowed = boundary.authorize_and_execute(
            _enforcement(
                _side_effect(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                ),
            ),
            lambda: calls.append("y") or "y",
        )
    assert isinstance(denied, MeaningfulSideEffectAuthorizationResult)
    assert denied.decision.action is PolicyAction.DENY
    assert allowed == "y"
    assert calls == ["y"]
