# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1-H2 — authoritative collaborative applicability closure tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
)
from intergrax.agent_distribution.coordination_binding_materialization import (
    CoordinationCollaborativeApplicabilityClassification,
    CoordinationCollaborativeApplicabilityIndeterminateError,
    classify_coordination_collaborative_applicability_from_governed_task,
    materialize_coordination_intent_binding,
    reconcile_coordination_collaborative_applicability,
)
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationIntentExecutor,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.collaborative_work import (
    MembershipStatus,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.multi_agent_coordination_governance import (
    MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,
    MultiAgentCoordinationCollaborativeApplicability,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
)
from intergrax.runtime.task.task import Task
from intergrax.agent_distribution.decision_coordination_projection import (
    project_authoritative_accepted_decision_coordination,
)
from intergrax.contracts.decision_coordination import DecisionCoordinationShape
from testing_support.agent_distribution.decision_coordination_qualification import (
    accepted_decision,
    build_decision_coordination_executor_fixture,
    decision_contribution,
)
from tests.unit.agent_distribution.test_coordination_intent import (
    _fan_out_intent,
    _single_intent,
)
from tests.unit.agent_distribution.test_coordination_intent_executor import (
    _StaticOrchestrationPort,
    _TrackingCoordinationService,
    _TrackingFanOutService,
    _binding,
    _success_outcome,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    _OCR_PACKAGE,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_delegated_subtasks import _discovery_candidate

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "admin-user"
_SCOPE = MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_ACTING,
        auth_subject="subject-acting",
    )


def _governed_task(
    *,
    workspace_id: str | None = _WORKSPACE,
    tenant_id: str = _TENANT,
    user_id: str = _ACTING,
) -> Task:
    metadata: dict[str, str] = {}
    if workspace_id is not None:
        metadata["workspace_id"] = workspace_id
    return Task(
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id="agent-a",
        metadata=metadata,
    )


def _resolver_with_membership() -> CollaborativeWorkAuthorityResolver:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
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
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-acting",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
        )
    )
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=delegation_repo,
        principal_authority_repository=authority_repo,
    )


def test_materialize_binding_from_governed_task_workspace() -> None:
    task_scope = mint_task_id()
    base = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    binding = materialize_coordination_intent_binding(
        task_scope_id=base.task_scope_id,
        application_id=base.application_id,
        application_environment_id=base.application_environment_id,
        contribution_bindings=base.contribution_bindings,
        governed_task=_governed_task(),
    )
    assert (
        binding.collaborative_applicability.applicability
        is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
    )
    assert binding.collaborative_applicability.workspace_id == _WORKSPACE


def test_classify_non_collaborative_without_governed_task() -> None:
    classification = classify_coordination_collaborative_applicability_from_governed_task(
        None,
    )
    assert (
        classification.applicability
        is MultiAgentCoordinationCollaborativeApplicability.NOT_APPLICABLE
    )


def test_indeterminate_workspace_context_fails_closed() -> None:
    task = _governed_task(workspace_id="   ")
    with pytest.raises(CoordinationCollaborativeApplicabilityIndeterminateError):
        classify_coordination_collaborative_applicability_from_governed_task(task)


def test_reconcile_omission_attack_fails_closed() -> None:
    authoritative_task = _governed_task()
    omitted = CoordinationCollaborativeApplicabilityClassification.not_applicable()
    reason = reconcile_coordination_collaborative_applicability(
        omitted,
        authoritative_task,
    )
    assert reason == "collaborative_applicability_authoritative_mismatch"


def test_reconcile_substitution_attack_fails_closed() -> None:
    authoritative_task = _governed_task(workspace_id="workspace-w1")
    substituted = CoordinationCollaborativeApplicabilityClassification.required(
        workspace_id="workspace-w2",
    )
    reason = reconcile_coordination_collaborative_applicability(
        substituted,
        authoritative_task,
    )
    assert reason == "collaborative_workspace_authoritative_mismatch"


def test_required_without_host_context_fails_closed() -> None:
    required = CoordinationCollaborativeApplicabilityClassification.required(
        workspace_id=_WORKSPACE,
    )
    reason = reconcile_coordination_collaborative_applicability(required, None)
    assert reason == "collaborative_applicability_without_authoritative_host_context"


@pytest.mark.asyncio
async def test_workspace_omission_attack_blocks_execution() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=AllowingMultiAgentCoordinationGovernance(authority_resolver=None),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        with pytest.raises(CoordinationGovernanceDenied) as exc_info:
            await executor.execute(intent, binding=binding, principal=admin_test_principal())
        assert (
            exc_info.value.result.decision.reason
            == "collaborative_applicability_authoritative_mismatch"
        )
    finally:
        governed.reset(token)
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_workspace_substitution_attack_blocks_execution() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=AllowingMultiAgentCoordinationGovernance(authority_resolver=None),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(
        task_scope,
        pairs=(("contrib-a", "lease-a"),),
        workspace_id="workspace-other",
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task(workspace_id=_WORKSPACE))
    try:
        with pytest.raises(CoordinationGovernanceDenied) as exc_info:
            await executor.execute(intent, binding=binding, principal=admin_test_principal())
        assert (
            exc_info.value.result.decision.reason
            == "collaborative_workspace_authoritative_mismatch"
        )
    finally:
        governed.reset(token)
    assert coordination.calls == 0


@pytest.mark.asyncio
async def test_true_non_collaborative_executes_without_authority_resolver() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=AllowingMultiAgentCoordinationGovernance(authority_resolver=None),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    result = await executor.execute(intent, binding=binding, principal=admin_test_principal())
    assert result.mode.value == "single"
    assert coordination.calls == 1


@pytest.mark.asyncio
async def test_collaborative_single_executes_with_authoritative_binding() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
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
    base = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    binding = materialize_coordination_intent_binding(
        task_scope_id=base.task_scope_id,
        application_id=base.application_id,
        application_environment_id=base.application_environment_id,
        contribution_bindings=base.contribution_bindings,
        governed_task=_governed_task(),
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        result = await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert result.mode.value == "single"
    assert coordination.calls == 1


@pytest.mark.asyncio
async def test_collaborative_fan_out_executes_with_authoritative_binding() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(
                (
                    _success_outcome("contrib-a", "a"),
                    _success_outcome("contrib-b", "b"),
                ),
            ),
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
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    base = _binding(
        task_scope,
        pairs=(("contrib-a", "lease-a"), ("contrib-b", "lease-b")),
    )
    binding = materialize_coordination_intent_binding(
        task_scope_id=base.task_scope_id,
        application_id=base.application_id,
        application_environment_id=base.application_environment_id,
        contribution_bindings=base.contribution_bindings,
        governed_task=_governed_task(),
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        result = await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert result.mode.value == "fan_out"
    assert fan_out.calls == 1


@pytest.mark.asyncio
async def test_decision_backed_path_omission_attack_fails_closed() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        fan_out=False,
    )
    decision = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-a"),),
    )
    intent = project_authoritative_accepted_decision_coordination(decision)
    task_scope = mint_task_id()
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        with pytest.raises(CoordinationGovernanceDenied) as exc_info:
            await fixture.executor.execute(
                intent,
                binding=binding,
                principal=admin_test_principal(),
            )
        assert (
            exc_info.value.result.decision.reason
            == "collaborative_applicability_authoritative_mismatch"
        )
    finally:
        governed.reset(token)
