# © Artur Czarnecki. All rights reserved.

"""GR-3 — canonical inner governance enforcement proofs."""

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
from intergrax.contracts.canonical_inner_governance import (
    assert_meaningful_side_effect_matches_active_execution,
    require_active_execution_for_meaningful_side_effect,
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
from intergrax.contracts.execution_identity import (
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

_TENANT = "tenant-gr3"
_WORKSPACE = "workspace-gr3"
_ACTING = "principal-gr3"
_OPERATION = "gr3.side_effect"
_SCOPE = "gr3.scope"
_RESOURCE = "resource-gr3"
_NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _seed_boundary(
    *,
    task_id: str,
    runtime_decision: PolicyAction = PolicyAction.ALLOW,
) -> MeaningfulSideEffectAuthorizationBoundary:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-gr3",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-gr3",
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
        runtime_policy_evaluator=RuntimePolicyEngine(
            meaningful_side_effect_rules=(
                MeaningfulSideEffectPolicyRule(
                    rule_id="runtime.gr3",
                    action=_OPERATION,
                    decision=runtime_decision,
                ),
            )
        ),
    )
    return MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=default_gr3_inner_guard(task_id),
    )


def _side_effect_request(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    execution_id: str,
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope-gr3",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id=_ACTING,
        tenant_id=_TENANT,
        resource=_RESOURCE,
    )


def _enforcement_request(
    side_effect: MeaningfulSideEffectRequest,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


def test_no_active_execution_blocks_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = _seed_boundary(task_id=task_id)
    request = _enforcement_request(
        _side_effect_request(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )
    calls: list[str] = []
    result = boundary.authorize_and_execute(request, lambda: calls.append("effect") or "ok")
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_exact_four_id_match_allow_executes_once() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = _seed_boundary(task_id=task_id)
    request = _enforcement_request(
        _side_effect_request(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            request,
            lambda: calls.append("effect") or "ok",
        )
    assert result == "ok"
    assert calls == ["effect"]


@pytest.mark.parametrize(
    ("mutator",),
    [
        (lambda ids: (mint_task_id(), ids[1], ids[2], ids[3]),),
        (lambda ids: (ids[0], mint_run_id(), ids[2], ids[3]),),
        (lambda ids: (ids[0], ids[1], mint_attempt_id(), ids[3]),),
        (lambda ids: (ids[0], ids[1], ids[2], mint_execution_id()),),
    ],
    ids=("task", "run", "attempt", "execution"),
)
def test_identity_mismatch_blocks_effect(mutator) -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = _seed_boundary(task_id=task_id)
    mismatched = mutator((task_id, run_id, attempt_id, execution_id))
    request = _enforcement_request(
        _side_effect_request(
            task_id=mismatched[0],
            run_id=mismatched[1],
            attempt_id=mismatched[2],
            execution_id=mismatched[3],
        ),
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(request, lambda: calls.append("effect"))
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_deny_and_require_human_execute_zero_effects() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    for decision in (PolicyAction.DENY, PolicyAction.REQUIRE_HUMAN):
        boundary = _seed_boundary(task_id=task_id, runtime_decision=decision)
        request = _enforcement_request(
            _side_effect_request(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        )
        calls: list[str] = []
        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ):
            result = boundary.authorize_and_execute(request, lambda: calls.append("effect"))
        assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
        assert calls == []


def test_policy_implementation_replaceable_without_consumer_change() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    request = _enforcement_request(
        _side_effect_request(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )
    allow_boundary = _seed_boundary(task_id=task_id, runtime_decision=PolicyAction.ALLOW)
    deny_boundary = _seed_boundary(task_id=task_id, runtime_decision=PolicyAction.DENY)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        allow_calls: list[str] = []
        deny_calls: list[str] = []
        allow_result = allow_boundary.authorize_and_execute(
            request,
            lambda: allow_calls.append("effect") or "ok",
        )
        deny_result = deny_boundary.authorize_and_execute(
            request,
            lambda: deny_calls.append("effect"),
        )
    assert allow_result == "ok"
    assert allow_calls == ["effect"]
    assert isinstance(deny_result, MeaningfulSideEffectAuthorizationResult)
    assert deny_calls == []


def test_contract_assert_four_id_match() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    request = _side_effect_request(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert_meaningful_side_effect_matches_active_execution(
        request,
        active_task_id=task_id,
        active_run_id=run_id,
        active_attempt_id=attempt_id,
        active_execution_id=execution_id,
    )


def test_contract_require_active_execution_for_side_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    request = _side_effect_request(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    scope = StaticActiveTaskScope(task_id)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        require_active_execution_for_meaningful_side_effect(request, task_scope=scope)
