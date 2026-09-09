# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1-H1 — collaborative authority + governance composition tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
)
from intergrax.agent_distribution.coordination_binding_materialization import (
    materialize_coordination_intent_binding,
)
from intergrax.agent_distribution.coordination_intent import CoordinationExecutionMode
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationGovernanceRequiresHuman,
    CoordinationIntentExecutor,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.collaborative_work import (
    DelegationStatus,
    MembershipStatus,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.multi_agent_coordination_governance import (
    MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,
    MultiAgentCoordinationCapabilityKind,
    MultiAgentCoordinationCollaborativeApplicability,
    MultiAgentCoordinationCollaborativeContext,
    MultiAgentCoordinationExecutionMode,
    MultiAgentCoordinationGovernanceContribution,
    MultiAgentCoordinationGovernancePolicyRule,
    MultiAgentCoordinationGovernanceRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
    DenyingMultiAgentCoordinationGovernance,
    MultiAgentCoordinationGovernanceBoundary,
    RequireHumanMultiAgentCoordinationGovernance,
    RuntimeMultiAgentCoordinationGovernance,
    _StaticMultiAgentCoordinationGovernanceEvaluator,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.task.task import Task
from tests.unit.agent_distribution.test_coordination_intent import _single_intent
from tests.unit.agent_distribution.test_coordination_intent_executor import (
    _StaticOrchestrationPort,
    _TrackingCoordinationService,
    _TrackingFanOutService,
    _binding,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "admin-user"
_DELEGATOR = "delegator-user"
_SCOPE = MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_ACTING,
        auth_subject="subject-acting",
    )


def _governed_task(*, workspace_id: str | None = _WORKSPACE) -> Task:
    metadata: dict[str, str] = {}
    if workspace_id is not None:
        metadata["workspace_id"] = workspace_id
    return Task(
        tenant_id=_TENANT,
        user_id=_ACTING,
        agent_id="agent-a",
        metadata=metadata,
    )


def _collaborative_binding(task_scope, *, pairs: tuple[tuple[str, str], ...]):
    base = _binding(task_scope, pairs=pairs)
    return materialize_coordination_intent_binding(
        task_scope_id=base.task_scope_id,
        application_id=base.application_id,
        application_environment_id=base.application_environment_id,
        contribution_bindings=base.contribution_bindings,
        governed_task=_governed_task(),
    )


def _request(
    *,
    collaborative: bool = False,
    delegator_principal_id: str | None = None,
    delegation_id: str | None = None,
) -> MultiAgentCoordinationGovernanceRequest:
    collaborative_applicability = (
        MultiAgentCoordinationCollaborativeApplicability.REQUIRED
        if collaborative
        else MultiAgentCoordinationCollaborativeApplicability.NOT_APPLICABLE
    )
    collaborative_context = None
    if collaborative:
        collaborative_context = MultiAgentCoordinationCollaborativeContext(
            workspace_id=_WORKSPACE,
            acting_principal_id=_ACTING,
            delegator_principal_id=delegator_principal_id,
            delegation_id=delegation_id,
        )
    return MultiAgentCoordinationGovernanceRequest(
        intent_id="intent-1",
        execution_mode=MultiAgentCoordinationExecutionMode.SINGLE,
        contributions=(
            MultiAgentCoordinationGovernanceContribution(
                contribution_id="contrib-a",
                capability_kind=MultiAgentCoordinationCapabilityKind.RESOLVED_REQUIREMENT,
                required_capability_ids=("document.ocr",),
            ),
        ),
        task_scope_id="task-scope-1",
        application_id="app-a",
        application_environment_id="env-a",
        principal=_principal(),
        collaborative_applicability=collaborative_applicability,
        collaborative_context=collaborative_context,
    )


def _allow_policy() -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="coordination_allow",
        policy_rule_id="test.coordination.allow",
    )


def _resolver_with_membership(
    *,
    membership_status: MembershipStatus = MembershipStatus.ACTIVE,
    authority_scopes: tuple[str, ...] = (_SCOPE,),
) -> CollaborativeWorkAuthorityResolver:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=membership_status,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-acting",
            principal_id=_ACTING,
            authority_scopes=authority_scopes,
        )
    )
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )


def _resolver_with_delegation(
    *,
    delegation_status: DelegationStatus = DelegationStatus.ACTIVE,
    delegation_scopes: tuple[str, ...] = (_SCOPE,),
) -> CollaborativeWorkAuthorityResolver:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-delegator",
            principal_id=_DELEGATOR,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-delegator",
            principal_id=_DELEGATOR,
            authority_scopes=delegation_scopes,
        )
    )
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=delegation_scopes,
            status=delegation_status,
        )
    )
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )


def _composed_boundary(
    *,
    policy_decision: PolicyDecision,
    resolver: CollaborativeWorkAuthorityResolver | None,
) -> MultiAgentCoordinationGovernanceBoundary:
    return MultiAgentCoordinationGovernanceBoundary(
        evaluator=_StaticMultiAgentCoordinationGovernanceEvaluator(policy_decision),
        authority_resolver=resolver,
    )


def test_request_identity_without_membership_denies_when_collaborative_required() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        clock=lambda: _NOW,
    )
    boundary = _composed_boundary(policy_decision=_allow_policy(), resolver=resolver)
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_valid_collaborative_authority_and_policy_allow() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW


def test_non_collaborative_path_does_not_require_authority_resolver() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=None,
    )
    result = boundary.evaluate(_request(collaborative=False))
    assert result.permitted is True


def test_stale_delegation_denies_even_when_policy_allows() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-delegator",
            principal_id=_DELEGATOR,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-delegator",
            principal_id=_DELEGATOR,
            authority_scopes=(_SCOPE,),
        )
    )
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=DelegationStatus.ACTIVE,
            valid_from=_NOW - timedelta(days=30),
            valid_until=_NOW - timedelta(seconds=1),
        )
    )
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )
    boundary = _composed_boundary(policy_decision=_allow_policy(), resolver=resolver)
    result = boundary.evaluate(
        _request(
            collaborative=True,
            delegator_principal_id=_DELEGATOR,
            delegation_id="delegation-1",
        ),
    )
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_revoked_membership_denies_even_when_policy_allows() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.REVOKED,
        )
    )
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        clock=lambda: _NOW,
    )
    boundary = _composed_boundary(policy_decision=_allow_policy(), resolver=resolver)
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_authority_non_amplification_denies() -> None:
    resolver = _resolver_with_membership(authority_scopes=("other.scope",))
    boundary = _composed_boundary(policy_decision=_allow_policy(), resolver=resolver)
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_policy_allow_cannot_override_authority_deny() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(membership_status=MembershipStatus.REVOKED),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_authority_allow_cannot_override_policy_deny() -> None:
    boundary = _composed_boundary(
        policy_decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="coordination_denied",
            policy_rule_id="test.coordination.deny",
        ),
        resolver=_resolver_with_membership(),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_require_human_with_authority_deny_remains_deny() -> None:
    boundary = _composed_boundary(
        policy_decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="coordination_require_human",
            policy_rule_id="test.coordination.require_human",
        ),
        resolver=_resolver_with_membership(membership_status=MembershipStatus.REVOKED),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.requires_governed_continuation is False
    assert result.decision.action is PolicyAction.DENY


def test_collaborative_required_without_resolver_fail_closed() -> None:
    boundary = _composed_boundary(policy_decision=_allow_policy(), resolver=None)
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.reason == "collaborative_authority_resolver_unavailable"


def test_runtime_governance_uses_policy_engine_facade_not_concrete_check() -> None:
    governance = RuntimeMultiAgentCoordinationGovernance(
        policy_engine=RuntimePolicyEngine(
            multi_agent_coordination_rules=(
                MultiAgentCoordinationGovernancePolicyRule(
                    rule_id="coordination.allow",
                    decision=PolicyAction.ALLOW,
                ),
            ),
        ),
        authority_resolver=_resolver_with_membership(),
    )
    result = governance.evaluate(_request(collaborative=True))
    assert result.permitted is True


@pytest.mark.asyncio
async def test_deny_has_zero_downstream_effects_with_collaborative_workspace() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    resolver = _resolver_with_membership(membership_status=MembershipStatus.REVOKED)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=DenyingMultiAgentCoordinationGovernance(authority_resolver=resolver),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _collaborative_binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        with pytest.raises(CoordinationGovernanceDenied):
            await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_collaborative_allow_executes_coordination_path() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=AllowingMultiAgentCoordinationGovernance(
            authority_resolver=_resolver_with_membership(),
        ),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _collaborative_binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        result = await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert result.mode is CoordinationExecutionMode.SINGLE
    assert coordination.calls == 1


@pytest.mark.asyncio
async def test_require_human_blocks_execution_before_approval() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=RequireHumanMultiAgentCoordinationGovernance(
            authority_resolver=_resolver_with_membership(),
        ),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _collaborative_binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        with pytest.raises(CoordinationGovernanceRequiresHuman):
            await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert coordination.calls == 0
    assert fan_out.calls == 0


def test_collaborative_allow_does_not_imply_execution_authority_expansion() -> None:
    """Governance ALLOW is semantic admission only — execution narrowing remains separate."""
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is True
    assert "ParentExecutionAuthority" not in result.model_dump_json()
    assert result.evidence.policy_action is PolicyAction.ALLOW
