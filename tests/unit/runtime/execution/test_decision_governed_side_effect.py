# © Artur Czarnecki. All rights reserved.

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
    DecisionGovernanceMaterialMismatchError,
    decision_governance_material_ref_from_accepted,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
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
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.decision_governed_side_effect import (
    DecisionGovernedSideEffectError,
    authorize_and_execute_decision_bound_side_effect,
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

_TENANT = "tenant-gr6"
_WORKSPACE = "workspace-gr6"
_ACTING = "principal-gr6"
_OPERATION = "gr6.side_effect"
_SCOPE = "gr6.scope"
_RESOURCE = "resource-gr6"
_NOW = datetime(2026, 9, 16, 8, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Payload:
    value: str


def _boundary(
    *,
    task_id: TaskId,
    runtime_decision: PolicyAction,
) -> MeaningfulSideEffectAuthorizationBoundary:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-gr6",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-gr6",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-gr6",
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
    gate = CollaborativeWorkEnforcementGate(
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
                    rule_id="runtime.gr6",
                    action=_OPERATION,
                    decision=runtime_decision,
                ),
            ),
        ),
    )
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=default_gr3_inner_guard(task_id),
    )


def _accepted_for_execution(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    return AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="gr6", subject="effect"),
            tenant_id=_TENANT,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr6.payload"),
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
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope-gr6",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_TENANT,
        principal_id=_ACTING,
        resource=_RESOURCE,
    )


def _enforcement(side_effect: MeaningfulSideEffectRequest) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


def _authorization_for(decision: AuthoritativeAcceptedDecision[_Payload]):
    action = decision_execution_action(kind=_OPERATION, subject=_RESOURCE)
    policy = decision_governance_policy_context(policy_provenance_digest="gr6-policy")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy,
        tenant_id=decision.identity.tenant_id,
    )
    return decision_execution_authorization(governance_decision=governance), action, policy


def test_decision_allow_platform_allow_executes_once() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted_for_execution(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision)
    boundary = _boundary(
        task_id=task_id,
        runtime_decision=PolicyAction.ALLOW,
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


def test_decision_allow_platform_deny_blocks_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted_for_execution(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision)
    boundary = _boundary(
        task_id=task_id,
        runtime_decision=PolicyAction.DENY,
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


def test_stale_decision_version_fails_closed() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision_v1 = _accepted_for_execution(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision_v1)
    v2 = next_decision_version(decision_v1.identity.version)
    decision_v2 = AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=decision_v1.identity.decision_id,
            version=v2,
            scope=decision_v1.identity.scope,
            tenant_id=decision_v1.identity.tenant_id,
            execution=decision_v1.identity.execution,
        ),
        artifact=decision_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(v2),
            parents=(decision_lineage_ref(decision_v1.identity.version),),
        ),
    )
    boundary = _boundary(
        task_id=task_id,
        runtime_decision=PolicyAction.ALLOW,
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        with pytest.raises(DecisionGovernedSideEffectError):
            authorize_and_execute_decision_bound_side_effect(
                boundary,
                enforcement_request=_enforcement(
                    _side_effect(
                        task_id=task_id,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        execution_id=execution_id,
                    ),
                ),
                decision=decision_v2,
                authorization=authorization,
                action=action,
                policy_context=policy,
                execute=lambda: "ok",
            )


def test_wrong_operation_action_mismatch_fails() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = _accepted_for_execution(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authorization, action, policy = _authorization_for(decision)
    wrong_action = decision_execution_action(kind="other.operation", subject=_RESOURCE)
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=wrong_action,
    )
    with pytest.raises(DecisionGovernanceMaterialMismatchError):
        from intergrax.contracts.decision_governance_material import (
            validate_decision_governance_material_for_decision,
        )

        validate_decision_governance_material_for_decision(
            material=material,
            decision=decision,
            action=action,
        )
