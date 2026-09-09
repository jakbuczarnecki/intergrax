# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1 Final — cross-layer governance admission qualification."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
)
from intergrax.agent_distribution.coordination_binding_materialization import (
    CoordinationCollaborativeApplicabilityClassification,
    classify_coordination_collaborative_applicability_from_governed_task,
    materialize_coordination_intent_binding,
    reconcile_coordination_collaborative_applicability,
)
from intergrax.agent_distribution.coordination_intent import CoordinationExecutionMode
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationGovernanceRequiresHuman,
    CoordinationIntentExecutor,
    CoordinationIntentResult,
)
from intergrax.agent_distribution.decision_coordination_projection import (
    project_authoritative_accepted_decision_coordination,
)
from intergrax.autonomous_work.execution_authority_admission import (
    CollaborativeWorkAuthorityResolverPort,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.collaborative_work import MembershipStatus
from intergrax.contracts.decision_coordination import DecisionCoordinationShape
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.contracts.multi_agent_coordination_governance import (
    MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,
    MultiAgentCoordinationCollaborativeApplicability,
    MultiAgentCoordinationGovernanceRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
    MultiAgentCoordinationGovernanceBoundary,
    _StaticMultiAgentCoordinationGovernanceEvaluator,
)
from intergrax.runtime.task.task import Task
from testing_support.agent_distribution.coordination_governance import (
    allowing_coordination_governance,
    bound_governed_host_task,
    denying_coordination_governance,
    non_collaborative_governed_host_task,
    require_human_coordination_governance,
)
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
    OcrRequest,
    OcrResult,
    _OCR_PACKAGE,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_delegated_subtasks import _discovery_candidate
from tests.unit.agent_distribution.test_multi_agent_coordination import _root_identity
from tests.unit.runtime.governance.test_multi_agent_coordination_collaborative_authority import (
    _allow_policy,
    _request,
    _resolver_with_membership,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "admin-user"
_SCOPE = MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE

_FORBIDDEN_ENGINE_NAMES = (
    "MultiAgentPolicyEngine",
    "CoordinationGovernanceEngine",
    "NpscGovernanceRuntime",
    "MultiAgentAuthorityEngine",
    "MultiAgentPolicyComposer",
)


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_ACTING,
        auth_subject="subject-acting",
    )


def _governed_task(
    *,
    workspace_id: str | None = _WORKSPACE,
) -> Task:
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


def _executor_bundle(
    *,
    governance,
    fan_out_items: tuple | None = None,
) -> tuple[
    CoordinationIntentExecutor[OcrRequest, OcrResult],
    _TrackingCoordinationService,
    _TrackingFanOutService,
]:
    coordination = _TrackingCoordinationService()
    outcomes = fan_out_items or (
        _success_outcome("contrib-a", "a"),
        _success_outcome("contrib-b", "b"),
    )
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(outcomes),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=cast(Any, coordination),
        fan_out=fan_out,
        governance=governance,
    )
    return executor, coordination, fan_out


async def _execute_decision_backed_intent(fixture, intent, binding) -> CoordinationIntentResult[OcrResult]:
    task_scope = binding.task_scope_id
    fixture.harness.task_scope_authority.task_scope_id = task_scope
    captured: list = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            with bound_governed_host_task(non_collaborative_governed_host_task()):
                result = await fixture.executor.execute(
                    intent,
                    binding=binding,
                    principal=admin_test_principal(),
                )
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0]


def _composed_boundary(
    *,
    policy_decision: PolicyDecision,
    resolver: CollaborativeWorkAuthorityResolver | None,
) -> MultiAgentCoordinationGovernanceBoundary:
    return MultiAgentCoordinationGovernanceBoundary(
        evaluator=_StaticMultiAgentCoordinationGovernanceEvaluator(policy_decision),
        authority_resolver=resolver,
    )


@pytest.mark.gate
def test_npsc5d_r1_no_second_governance_engine() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        if "build" in path.parts or ".tmp" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name in _FORBIDDEN_ENGINE_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_r1_port_ownership_shared_consumer_seam() -> None:
    port_module = (
        _REPO_ROOT
        / "intergrax"
        / "autonomous_work"
        / "execution_authority_admission.py"
    )
    source = port_module.read_text(encoding="utf-8-sig")
    assert "class CollaborativeWorkAuthorityResolverPort" in source
    assert "AW-3B" in source
    assert issubclass(CollaborativeWorkAuthorityResolverPort, object)


@pytest.mark.gate
def test_npsc5d_r1_npsc_delegation_distinct_from_authority_delegation() -> None:
    delegated_path = _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py"
    authority_path = _REPO_ROOT / "intergrax" / "collaborative_work" / "authority.py"
    delegated_source = delegated_path.read_text(encoding="utf-8-sig")
    authority_source = authority_path.read_text(encoding="utf-8-sig")
    assert "AuthorityDelegation" not in delegated_source
    assert "DelegatedSubtaskService" not in authority_source
    assert "ChildExecution" not in authority_source


@pytest.mark.gate
def test_npsc5d_r1_request_identity_distinct_from_authority_and_policy() -> None:
    principal = _principal()
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert isinstance(principal, RequestIdentity)
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW
    assert type(principal).__name__ != type(result.decision).__name__


@pytest.mark.gate
def test_npsc5d_r1_policy_composition_precedence() -> None:
    deny_policy = PolicyDecision(
        action=PolicyAction.DENY,
        reason="coordination_denied",
        policy_rule_id="test.coordination.deny",
    )
    require_human = PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="coordination_require_human",
        policy_rule_id="test.coordination.require_human",
    )
    revoked = _resolver_with_membership(membership_status=MembershipStatus.REVOKED)
    valid = _resolver_with_membership()

    authority_deny_policy_allow = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=revoked,
    ).evaluate(_request(collaborative=True))
    assert authority_deny_policy_allow.decision.action is PolicyAction.DENY

    authority_allow_policy_deny = _composed_boundary(
        policy_decision=deny_policy,
        resolver=valid,
    ).evaluate(_request(collaborative=True))
    assert authority_allow_policy_deny.decision.action is PolicyAction.DENY

    authority_allow_require_human = _composed_boundary(
        policy_decision=require_human,
        resolver=valid,
    ).evaluate(_request(collaborative=True))
    assert authority_allow_require_human.decision.action is PolicyAction.REQUIRE_HUMAN

    authority_deny_require_human = _composed_boundary(
        policy_decision=require_human,
        resolver=revoked,
    ).evaluate(_request(collaborative=True))
    assert authority_deny_require_human.decision.action is PolicyAction.DENY


@pytest.mark.gate
def test_npsc5d_r1_policy_allow_does_not_expand_execution_authority() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is True
    assert "ParentExecutionAuthority" not in result.model_dump_json()


@pytest.mark.gate
def test_npsc5d_r1_authoritative_applicability_semantics() -> None:
    required = classify_coordination_collaborative_applicability_from_governed_task(
        _governed_task(),
    )
    assert (
        required.applicability
        is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
    )

    not_applicable = classify_coordination_collaborative_applicability_from_governed_task(
        _governed_task(workspace_id=None),
    )
    assert (
        not_applicable.applicability
        is MultiAgentCoordinationCollaborativeApplicability.NOT_APPLICABLE
    )

    omission_reason = reconcile_coordination_collaborative_applicability(
        CoordinationCollaborativeApplicabilityClassification.not_applicable(),
        _governed_task(),
    )
    assert omission_reason == "collaborative_applicability_authoritative_mismatch"

    missing_host = reconcile_coordination_collaborative_applicability(
        CoordinationCollaborativeApplicabilityClassification.required(
            workspace_id=_WORKSPACE,
        ),
        None,
    )
    assert missing_host == "collaborative_applicability_without_authoritative_host_context"


@pytest.mark.asyncio
async def test_npsc5d_r1_single_non_collaborative_allow() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    with bound_governed_host_task(non_collaborative_governed_host_task()):
        result = await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )
    assert result.mode is CoordinationExecutionMode.SINGLE
    assert coordination.calls == 1
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_single_collaborative_allow() -> None:
    executor, coordination, fan_out = _executor_bundle(
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
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_single_collaborative_authority_deny() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=AllowingMultiAgentCoordinationGovernance(
            authority_resolver=_resolver_with_membership(
                membership_status=MembershipStatus.REVOKED,
            ),
        ),
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
async def test_npsc5d_r1_fan_out_non_collaborative_allow() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    binding = _binding(
        task_scope,
        pairs=(("contrib-a", "lease-a"), ("contrib-b", "lease-b")),
    )
    with bound_governed_host_task(non_collaborative_governed_host_task()):
        result = await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )
    assert result.mode is CoordinationExecutionMode.FAN_OUT
    assert fan_out.calls == 1
    assert coordination.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_fan_out_collaborative_allow() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=AllowingMultiAgentCoordinationGovernance(
            authority_resolver=_resolver_with_membership(),
        ),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    binding = _collaborative_binding(
        task_scope,
        pairs=(("contrib-a", "lease-a"), ("contrib-b", "lease-b")),
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(_governed_task())
    try:
        result = await executor.execute(intent, binding=binding, principal=_principal())
    finally:
        governed.reset(token)
    assert result.mode is CoordinationExecutionMode.FAN_OUT
    assert fan_out.calls == 1
    assert coordination.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_fan_out_governance_deny() -> None:
    orchestration = _StaticOrchestrationPort(
        (
            _success_outcome("contrib-a", "a"),
            _success_outcome("contrib-b", "b"),
        ),
    )
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=orchestration),
    )
    coordination = _TrackingCoordinationService()
    executor = CoordinationIntentExecutor(
        coordination=cast(Any, coordination),
        fan_out=fan_out,
        governance=denying_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    binding = _binding(
        task_scope,
        pairs=(("contrib-a", "lease-a"), ("contrib-b", "lease-b")),
    )
    with bound_governed_host_task():
        with pytest.raises(CoordinationGovernanceDenied):
            await executor.execute(
                intent,
                binding=binding,
                principal=admin_test_principal(),
            )
    assert fan_out.calls == 0
    assert orchestration.calls == 0
    assert coordination.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_missing_governed_context_deny() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    with pytest.raises(CoordinationGovernanceDenied) as exc_info:
        await executor.execute(intent, binding=binding, principal=admin_test_principal())
    assert (
        exc_info.value.result.decision.reason
        == "collaborative_applicability_without_authoritative_host_context"
    )
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_workspace_omission_attack_deny() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=allowing_coordination_governance(),
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
async def test_npsc5d_r1_workspace_substitution_attack_deny() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=allowing_coordination_governance(),
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
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_require_human_blocks_downstream() -> None:
    executor, coordination, fan_out = _executor_bundle(
        governance=require_human_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    with bound_governed_host_task():
        with pytest.raises(CoordinationGovernanceRequiresHuman):
            await executor.execute(
                intent,
                binding=binding,
                principal=admin_test_principal(),
            )
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_npsc5d_r1_decision_backed_path_same_boundary() -> None:
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
    result = await _execute_decision_backed_intent(fixture, intent, binding)
    assert result.mode is CoordinationExecutionMode.SINGLE


@pytest.mark.asyncio
async def test_npsc5d_r1_deterministic_producer_same_boundary() -> None:
    executor, coordination, fan_out = _executor_bundle(
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


@pytest.mark.gate
def test_npsc5d_r1_governance_request_has_no_physical_agent_identity() -> None:
    request = _request(collaborative=True)
    assert "agent_id" not in MultiAgentCoordinationGovernanceRequest.model_fields
    assert "lease_id" not in MultiAgentCoordinationGovernanceRequest.model_fields
    assert request.collaborative_applicability is (
        MultiAgentCoordinationCollaborativeApplicability.REQUIRED
    )


@pytest.mark.gate
def test_npsc5d_r1_authority_non_amplification() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(authority_scopes=("other.scope",)),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


@pytest.mark.gate
def test_npsc5d_r1_revoked_membership_denies() -> None:
    boundary = _composed_boundary(
        policy_decision=_allow_policy(),
        resolver=_resolver_with_membership(membership_status=MembershipStatus.REVOKED),
    )
    result = boundary.evaluate(_request(collaborative=True))
    assert result.permitted is False


@pytest.mark.gate
def test_npsc5d_r1_adapter_public_contract_dependency_only() -> None:
    adapter_path = (
        _REPO_ROOT
        / "intergrax"
        / "agent_distribution"
        / "coordination_governance_adapter.py"
    )
    tree = ast.parse(adapter_path.read_text(encoding="utf-8-sig"), filename=str(adapter_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = [
        module
        for module in sorted(modules)
        if module.startswith("intergrax.runtime.governance")
        or module.startswith("intergrax.runtime.policy.runtime_policy_engine")
    ]
    assert forbidden == []
