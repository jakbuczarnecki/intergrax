# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-3 — governed operator checkpoint resume proofs (TASKCPM-R1–R24)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.agent_distribution.control_plane_governance import build_activation_mutation_request
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    build_harness_control_plane_governance,
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_host_control_plane_policy_bundle,
)
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.task_control import (
    HitlResumeValidationError,
    TaskControlValidationError,
    governed_resume_checkpoint_task,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    MUTATION_TYPE_RESUME_TASK_EXECUTION,
    MUTATION_TYPE_SET_TASK_AUTONOMY,
    TASK_EXECUTION_RESOURCE_TYPE,
    TaskControlGovernanceBlockedError,
    build_resume_task_execution_mutation_request,
    task_checkpoint_resume_current_revision,
    task_execution_resource_id,
    task_execution_resource_scope,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import RuntimePolicyBundleEvaluator
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-task-control"
_OTHER_TENANT = "tenant-other"
_MUTATION_ID = "mut-resume-1"
_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_ATTEMPT_ID = mint_attempt_id()
_PAUSE_ID = "pause-1"
_HUMAN_REQUEST_ID = "hr-1"
_RESUME_TOKEN = "resume-token-1"
_T0 = build_harness_host_control_plane_policy_bundle().issued_at


class _FakeIdentityProvider:
    def verify_token(self, token: str) -> IdentityUser:
        if token != "valid-bearer":
            raise ValueError("invalid")
        return IdentityUser(user_id="operator-1", tenant_id=_TENANT)


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.test_allow",
            decision_id="dec-allow",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)
    on_evaluate: object | None = None

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.on_evaluate is not None:
            self.on_evaluate()
        return self.decision


def _principal(*, tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="operator-1",
        principal_type=PrincipalType.USER,
        auth_subject="operator-1",
    )


def _paused_task(*, tenant_id: str = _TENANT) -> Task:
    task = Task(
        tenant_id=tenant_id,
        user_id="task-owner",
        message="paused work",
        task_id=_TASK_ID,
    )
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=_PAUSE_ID,
        task_id=_TASK_ID,
        human_request_id=_HUMAN_REQUEST_ID,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=_HUMAN_REQUEST_ID,
        prompt="approve?",
    )
    return task


def _checkpoint(
    *,
    tenant_id: str = _TENANT,
    checkpoint_id: str = "chk-1",
    resume_token: str = _RESUME_TOKEN,
    task_state: TaskState = TaskState.WAITING_FOR_HUMAN,
) -> TaskCheckpoint:
    task = _paused_task(tenant_id=tenant_id)
    return TaskCheckpoint(
        checkpoint_id=checkpoint_id,
        task_id=str(_TASK_ID),
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_snapshot=task.model_dump(mode="json"),
        task_state=task_state,
        progress_message="paused",
        notify_channel="debug",
        runtime=RuntimeCheckpoint(
            run_id=str(_RUN_ID),
            attempt_id=str(_ATTEMPT_ID),
        ),
    )


def _task_result() -> TaskResult:
    return TaskResult(
        task_id=str(_TASK_ID),
        state=TaskState.COMPLETED,
        answer="done",
    )


def _allow_boundary(
    *,
    on_evaluate: object | None = None,
) -> tuple[ControlPlaneMutationAuthorizationBoundary, _RecordingEvaluator]:
    evaluator = _RecordingEvaluator(on_evaluate=on_evaluate)
    return ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator), evaluator


class _StaticCheckpointStore:
    def __init__(self, checkpoint: TaskCheckpoint | None) -> None:
        self._checkpoint = checkpoint

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        if self._checkpoint is None:
            return None
        if (
            task_id == self._checkpoint.task_id
            and tenant_id == self._checkpoint.tenant_id
            and resume_token == self._checkpoint.resume_token
        ):
            return self._checkpoint
        return None


class _StaleAfterAllowCheckpointStore(TaskCheckpointPersistence):
    def __init__(self, checkpoint: TaskCheckpoint, *, stale: TaskCheckpoint | None) -> None:
        self._initial = checkpoint
        self._stale = stale
        self._lookup_count = 0

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        self._lookup_count += 1
        if self._lookup_count == 1:
            return self._initial
        if self._stale is None:
            return None
        return self._stale

    def list_for_task(self, task_id: str, tenant_id: str) -> list[TaskCheckpoint]:
        return []

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        return None

    def list_paused(self) -> list[TaskCheckpoint]:
        return []

    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        return checkpoint


@pytest.mark.asyncio
async def test_taskcpm_r1_allow_exact_checkpoint_invokes_runner_once() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={"verdict": "approve"},
            approver=local_development_approver_evidence(tenant_id=_TENANT),
        )
    assert outcome.accepted is True
    assert resume_call.await_count == 1
    assert resume_call.await_args.kwargs["checkpoint"].checkpoint_id == checkpoint.checkpoint_id
    assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_taskcpm_r2_caller_mutation_id_preserved() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )
    assert evaluator.calls[0].mutation_id == _MUTATION_ID


@pytest.mark.asyncio
async def test_taskcpm_r3_mutation_type_is_resume_task_execution() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )
    assert evaluator.calls[0].mutation_type == MUTATION_TYPE_RESUME_TASK_EXECUTION


@pytest.mark.asyncio
async def test_taskcpm_r4_task_id_bound_from_route_and_checkpoint() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )
    request = evaluator.calls[0]
    assert request.task_id == _TASK_ID
    assert request.resource_id == task_execution_resource_id(task_id=_TASK_ID, run_id=_RUN_ID)


@pytest.mark.asyncio
async def test_taskcpm_r5_run_id_from_checkpoint_canonical_identity() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )
    assert evaluator.calls[0].run_id == _RUN_ID


@pytest.mark.asyncio
async def test_taskcpm_r6_wrong_tenant_zero_runner_invocation() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(tenant_id=_OTHER_TENANT),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "tenant_authority_mismatch"
    assert resume_call.await_count == 0
    assert boundary.evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_r7_deny_zero_runner_invocation() -> None:
    checkpoint = _checkpoint()
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny_resume",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={"verdict": "approve"},
            approver=local_development_approver_evidence(tenant_id=_TENANT),
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.blocker_code == "TASK_CONTROL_BLOCKED_BY_POLICY"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r8_require_human_zero_runner_invocation() -> None:
    checkpoint = _checkpoint()
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="approval_required",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.require_human",
            decision_id="dec-human",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={"verdict": "approve"},
            approver=local_development_approver_evidence(tenant_id=_TENANT),
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.blocker_code == "TASK_CONTROL_BLOCKED_BY_REQUIRE_HUMAN"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r9_missing_boundary_raises_without_runner() -> None:
    checkpoint = _checkpoint()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        with pytest.raises(TaskControlGovernanceBlockedError) as exc_info:
            await governed_resume_checkpoint_task(
                runner,
                task_id=str(_TASK_ID),
                tenant_id=_TENANT,
                resume_token=_RESUME_TOKEN,
                mutation_id=_MUTATION_ID,
                principal=_principal(),
                mutation_boundary=None,
                checkpoint_store=_StaticCheckpointStore(checkpoint),
            )
    assert exc_info.value.blocker_code == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert resume_call.await_count == 0


def test_taskcpm_r10_cancel_rule_does_not_authorize_resume() -> None:
    cancel_only_bundle = build_immutable_runtime_policy_bundle(
        bundle_id="harness.control_plane",
        version="1.0.0-test",
        rules=(
            PolicyBundleRule(
                rule_id="harness.task_control.cancel_task_execution",
                match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
                effect="allow",
            ),
        ),
        issued_at=_T0,
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(cancel_only_bundle),
        ),
    )
    request = build_resume_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        mutation_id=_MUTATION_ID,
        checkpoint=_checkpoint(),
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_taskcpm_r11_autonomy_rule_does_not_authorize_resume() -> None:
    autonomy_only_bundle = build_immutable_runtime_policy_bundle(
        bundle_id="harness.control_plane",
        version="1.0.0-test",
        rules=(
            PolicyBundleRule(
                rule_id="harness.task_control.set_task_autonomy",
                match_action=MUTATION_TYPE_SET_TASK_AUTONOMY,
                effect="allow",
            ),
        ),
        issued_at=_T0,
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(autonomy_only_bundle),
        ),
    )
    result = boundary.authorize(
        build_resume_task_execution_mutation_request(
            principal=_principal(),
            tenant_id=_TENANT,
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            mutation_id=_MUTATION_ID,
            checkpoint=_checkpoint(),
        )
    )
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_taskcpm_r12_unrelated_mutation_not_authorized_by_resume_rule() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    lifecycle_request = build_activation_mutation_request(
        principal=_principal(),
        application_id="app-1",
        application_environment_id="env-1",
        mutation_id="mut-activate",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-1",
    )
    result = boundary.authorize(lifecycle_request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


@pytest.mark.asyncio
async def test_taskcpm_r13_checkpoint_disappears_after_allow_zero_runner() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    store = _StaleAfterAllowCheckpointStore(checkpoint, stale=None)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=store,
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "stale_checkpoint"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r14_checkpoint_identity_changes_after_allow_zero_runner() -> None:
    checkpoint = _checkpoint()
    replaced = _checkpoint(checkpoint_id="chk-replaced")
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    store = _StaleAfterAllowCheckpointStore(checkpoint, stale=replaced)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=store,
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "stale_checkpoint"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r15_checkpoint_run_id_conflict_zero_runner() -> None:
    checkpoint = _checkpoint()
    conflicting = _checkpoint()
    conflicting.runtime = RuntimeCheckpoint(
        run_id=str(mint_run_id()),
        attempt_id=str(_ATTEMPT_ID),
    )
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    store = _StaleAfterAllowCheckpointStore(checkpoint, stale=conflicting)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=store,
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "stale_checkpoint"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r16_resume_token_stale_zero_runner() -> None:
    checkpoint = _checkpoint()
    replaced = _checkpoint(resume_token="resume-token-replaced")
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    store = _StaleAfterAllowCheckpointStore(checkpoint, stale=replaced)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=store,
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "stale_checkpoint"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r17_hitl_pause_id_anti_forgery_still_enforced() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with pytest.raises(HitlResumeValidationError, match="pause_id conflicts"):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={
                "verdict": "approve",
                "pause_id": "FORGED",
            },
            approver=local_development_approver_evidence(tenant_id=_TENANT),
        )


@pytest.mark.asyncio
async def test_taskcpm_r18_execution_hitl_verdict_does_not_bypass_cpm_deny() -> None:
    checkpoint = _checkpoint()
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny_resume",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={"verdict": "approve"},
            approver=local_development_approver_evidence(tenant_id=_TENANT),
        )
    assert outcome.accepted is False
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r19_cpm_allow_does_not_fabricate_missing_hitl_verdict() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={},
            approver=None,
        )
    assert outcome.accepted is True
    assert resume_call.await_count == 1
    passed_operator_input = resume_call.await_args.kwargs["operator_input"]
    assert passed_operator_input == {}
    assert resume_call.await_args.kwargs["approver"] is None


@pytest.mark.asyncio
async def test_taskcpm_r20_http_caller_request_identity_preserved() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ):
        mount_harness_task_routes(
            app,
            task_runner=runner,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            mutation_boundary=boundary,
        )
        client = TestClient(app)
        response = client.post(
            f"/v1/tasks/{_TASK_ID}/resume",
            json={
                "mutation_id": _MUTATION_ID,
                "resume_token": _RESUME_TOKEN,
                "operator_input": {"verdict": "approve"},
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 200
    assert evaluator.calls[0].principal.tenant_id == _TENANT
    assert evaluator.calls[0].principal.user_id == "operator-1"


def test_taskcpm_r21_product_host_uses_canonical_bundle_authority() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    result = boundary.authorize(
        build_resume_task_execution_mutation_request(
            principal=_principal(),
            tenant_id=_TENANT,
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            mutation_id=_MUTATION_ID,
            checkpoint=_checkpoint(),
        )
    )
    assert result.permitted is True
    assert result.decision.policy_rule_id == "harness.task_control.resume_task_execution"


def test_taskcpm_r22_lab_without_boundary_remains_fail_closed() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is None


def test_taskcpm_r23_scheduler_resume_path_unchanged() -> None:
    from intergrax.runtime.long_running.scheduler import LongRunningScheduler

    source = LongRunningScheduler.__module__
    assert "governed_resume_checkpoint_task" not in source


def test_taskcpm_r24_supported_route_requires_governance_boundary() -> None:
    checkpoint = _checkpoint()
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=_task_result(),
    ) as resume_call:
        mount_harness_task_routes(
            app,
            task_runner=runner,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            mutation_boundary=None,
        )
        client = TestClient(app)
        response = client.post(
            f"/v1/tasks/{_TASK_ID}/resume",
            json={
                "mutation_id": _MUTATION_ID,
                "resume_token": _RESUME_TOKEN,
                "operator_input": {"verdict": "approve"},
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 403
    assert response.json()["detail"]["blocker_code"] == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert resume_call.await_count == 0


@pytest.mark.asyncio
async def test_taskcpm_r_revision_binds_checkpoint_identity() -> None:
    checkpoint = _checkpoint()
    request = build_resume_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        mutation_id=_MUTATION_ID,
        checkpoint=checkpoint,
    )
    assert request.current_revision == task_checkpoint_resume_current_revision(checkpoint)
    assert request.resource_scope == task_execution_resource_scope(
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
    )
    assert request.resource_type == TASK_EXECUTION_RESOURCE_TYPE


@pytest.mark.asyncio
async def test_taskcpm_r_missing_mutation_id_raises() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with pytest.raises(TaskControlValidationError, match="mutation_id_required"):
        await governed_resume_checkpoint_task(
            runner,
            task_id=str(_TASK_ID),
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id="   ",
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
        )


def test_taskcpm_r_http_unauthenticated_returns_401() -> None:
    checkpoint = _checkpoint()
    boundary, _ = _allow_boundary()
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    mount_harness_task_routes(
        app,
        task_runner=AsyncMock(spec=UnifiedTaskRunner),
        checkpoint_store=_StaticCheckpointStore(checkpoint),
        mutation_boundary=boundary,
    )
    client = TestClient(app)
    response = client.post(
        f"/v1/tasks/{_TASK_ID}/resume",
        json={
            "mutation_id": _MUTATION_ID,
            "resume_token": _RESUME_TOKEN,
        },
    )
    assert response.status_code == 401
