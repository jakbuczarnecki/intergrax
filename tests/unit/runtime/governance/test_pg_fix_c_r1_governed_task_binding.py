# © Artur Czarnecki. All rights reserved.

"""PG-FIX-C R1 — trusted runtime Task carrier (no metadata authority path)."""

from __future__ import annotations

import asyncio
import importlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.steps.domain_job import run_domain_job
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
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
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import (
    mint_task_id,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
    current_governed_execution_task,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.task.task import Task, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORGED_METADATA_KEY = "runtime.governed_execution.task.v1"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_TASK_ID = mint_task_id()
_RUN_ID = validate_run_id("run_" + ("ab" * 16))
_ATTEMPT_ID = validate_attempt_id("attempt_" + ("cd" * 16))
_PRINCIPAL = "principal-pg-c-r1"
_DIGEST = "sha256:" + ("cd" * 32)
_IDEM = "idem-pg-fix-c-r1"
_POLICY_RULE = "runtime.hitl"
_BUNDLE_ID = "bundle-pg-c-r1"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_REPO_ROOT = Path(__file__).resolve().parents[4]


class MutableRuntimePolicyEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        return self._decision


class _RecordingIntegration(DeterministicExternalWorkFake):
    def __init__(self, *, call_log: list[str] | None = None) -> None:
        super().__init__()
        self.call_log = call_log if call_log is not None else []

    def create_work(self, request):  # type: ignore[no-untyped-def]
        self.call_log.append("integration.create_work")
        return super().create_work(request)


def _decision(*, action: PolicyAction = PolicyAction.REQUIRE_HUMAN) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="pg-fix-c-r1-test",
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
    )


def _grant() -> GovernedContinuationApprovalGrant:
    return GovernedContinuationApprovalGrant.model_validate(
        {
            "grant_id": "gcg_pg_fix_c_r1",
            "continuation_request_id": "gcr_pg_fix_c_r1",
            "side_effect_scope_id": _IDEM,
            "side_effect_scope_digest": _DIGEST,
            "task_id": _TASK_ID,
            "run_id": _RUN_ID,
            "operation_id": ACTION_CREATE_EXTERNAL_WORK,
            "resource_scope": _DIGEST,
            "policy_rule_id": _POLICY_RULE,
            "policy_bundle_id": _BUNDLE_ID,
            "policy_bundle_version": _BUNDLE_V1,
            "policy_bundle_digest": _BUNDLE_D1,
            "pause_id": "pause-pg-c-r1",
            "human_request_id": "hr-pg-c-r1",
            "approved_at": "2026-08-19T00:00:00+00:00",
        }
    )


def _task() -> Task:
    return Task(tenant_id=_TENANT, user_id=_PRINCIPAL, message="x", task_id=_TASK_ID)


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _IDEM,
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }


def _seed_boundary() -> MeaningfulSideEffectAuthorizationBoundary:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-pg-c-r1",
            principal_id=_PRINCIPAL,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-pg-c-r1",
            principal_id=_PRINCIPAL,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=ACTION_CREATE_EXTERNAL_WORK,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
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
        runtime_policy_evaluator=MutableRuntimePolicyEvaluator(_decision()),
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)


def _step_ctx(
    task: Task,
    *,
    forged_task: Task | None = None,
    exec_ctx: RuntimeExecutionContext | None = None,
) -> AgentStepContext:
    request_metadata = dict(_meta())
    if forged_task is not None:
        request_metadata[_FORGED_METADATA_KEY] = forged_task
    request = AgentRunRequest(
        input="review PR",
        identity=RequestIdentity(tenant_id=_TENANT, user_id=_PRINCIPAL),
        metadata=request_metadata,
    )
    runtime_exec_ctx = exec_ctx or RuntimeExecutionContext(
        task_id=task.task_id,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        agent_id="external_contractor_adapter",
        request=request,
    )
    return AgentStepContext(
        task_id=str(task.task_id),
        run_id=str(_RUN_ID),
        tenant_id=_TENANT,
        message="review PR",
        metadata={**_meta(), "uaep_exec_ctx": runtime_exec_ctx},
    )


def test_r1_no_task_metadata_carrier_in_production() -> None:
    module = importlib.import_module(
        "intergrax.runtime.policy.meaningful_side_effect_authorization",
    )
    assert not hasattr(module, "GOVERNED_EXECUTION_TASK_METADATA_KEY")
    assert not hasattr(module, "resolve_governed_execution_task")
    graph_src = (_REPO_ROOT / "intergrax/runtime/nexus/execution/graph_executor.py").read_text(
        encoding="utf-8",
    )
    assert _FORGED_METADATA_KEY not in graph_src
    assert "GOVERNED_EXECUTION_TASK_METADATA_KEY" not in graph_src


@pytest.mark.asyncio
async def test_r2_forged_metadata_cannot_authorize() -> None:
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    log: list[str] = []
    boundary = _seed_boundary()
    step_ctx = _step_ctx(task, forged_task=task)
    result = await run_domain_job(
        step_ctx,
        external_work=_RecordingIntegration(call_log=log),
        authorization_boundary=boundary,
    )
    assert result["domain_summary"]["used"] is False
    assert log == []


@pytest.mark.asyncio
async def test_r3_trusted_binding_executes_once() -> None:
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    log: list[str] = []
    boundary = _seed_boundary()
    binding = ActiveGovernedExecutionTask()
    token = binding.bind(task)
    try:
        result = await run_domain_job(
            _step_ctx(task),
            external_work=_RecordingIntegration(call_log=log),
            authorization_boundary=boundary,
        )
    finally:
        binding.reset(token)
    assert result["domain_summary"]["used"] is True
    assert log == ["integration.create_work"]
    assert task.runtime.governance.governed_continuation_grant is None


@pytest.mark.asyncio
async def test_r4_boundary_receives_exact_bound_task_instance() -> None:
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    seen: list[Task | None] = []
    boundary = _seed_boundary()
    original = boundary.authorize_and_execute

    def _capture(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs.get("task"))
        return original(*args, **kwargs)

    boundary.authorize_and_execute = _capture  # type: ignore[method-assign]
    binding = ActiveGovernedExecutionTask()
    token = binding.bind(task)
    try:
        await run_domain_job(
            _step_ctx(task),
            external_work=_RecordingIntegration(),
            authorization_boundary=boundary,
        )
    finally:
        binding.reset(token)
    assert len(seen) == 1
    assert seen[0] is task


def test_r5_context_cleanup_on_success() -> None:
    task = _task()
    binding = ActiveGovernedExecutionTask()
    token = binding.bind(task)
    binding.reset(token)
    assert current_governed_execution_task() is None


def test_r5b_context_cleanup_on_exception() -> None:
    task = _task()
    binding = ActiveGovernedExecutionTask()
    token = binding.bind(task)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            raise RuntimeError("boom")
    finally:
        binding.reset(token)
    assert current_governed_execution_task() is None


@pytest.mark.asyncio
async def test_r6_concurrency_isolation() -> None:
    task_a = _task()
    task_b = Task(
        tenant_id=_TENANT,
        user_id=_PRINCIPAL,
        message="y",
        task_id=mint_task_id(),
    )
    binding = ActiveGovernedExecutionTask()
    barrier = asyncio.Barrier(2)

    async def _run(task: Task) -> str:
        token = binding.bind(task)
        try:
            await barrier.wait()
            await asyncio.sleep(0)
            current = current_governed_execution_task()
            assert current is not None
            return str(current.task_id)
        finally:
            binding.reset(token)

    seen_a, seen_b = await asyncio.gather(_run(task_a), _run(task_b))
    assert seen_a == str(task_a.task_id)
    assert seen_b == str(task_b.task_id)


def test_r7_agent_run_request_serializes_without_task_leak() -> None:
    request = AgentRunRequest(
        input="review PR",
        identity=RequestIdentity(tenant_id=_TENANT, user_id=_PRINCIPAL),
        metadata=dict(_meta()),
    )
    payload = request.model_dump_json()
    assert _FORGED_METADATA_KEY not in payload
    assert "Task(" not in payload


@pytest.mark.asyncio
async def test_r11_checkpoint_resume_uses_binding_not_metadata() -> None:
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        resume_token="rt-pg-c-r1",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
    )
    restored = Task.model_validate(checkpoint.task_snapshot)
    assert restored.runtime.governance.governed_continuation_grant is not None
    log: list[str] = []
    boundary = _seed_boundary()
    binding = ActiveGovernedExecutionTask()
    token = binding.bind(restored)
    try:
        result = await run_domain_job(
            _step_ctx(restored, forged_task=restored),
            external_work=_RecordingIntegration(call_log=log),
            authorization_boundary=boundary,
        )
    finally:
        binding.reset(token)
    assert result["domain_summary"]["used"] is True
    assert log == ["integration.create_work"]
    assert restored.runtime.governance.governed_continuation_grant is None
