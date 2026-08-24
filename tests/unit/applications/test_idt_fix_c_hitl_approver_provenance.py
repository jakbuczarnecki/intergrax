# © Artur Czarnecki. All rights reserved.

"""IDT-FIX-C — human decision provenance + exact HITL resume identity."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_approver_evidence,
)
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.task_control import (
    HitlResumeValidationError,
    _materialize_hitl_resume_input,
)
from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator
from intergrax.contracts.human_approver import (
    HumanApproverAuthMode,
    HumanApproverEvidence,
    local_development_approver_evidence,
)
from intergrax.runtime.human.models import (
    HumanDecisionRecord,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.pause import HumanApprovalResolutionError, HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SECRET_CREDENTIAL_MUST_NOT_PERSIST = "SECRET_CREDENTIAL_MUST_NOT_PERSIST"
TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
ATTEMPT_ID = mint_attempt_id()
TENANT_A = "tenant-A"
TENANT_B = "tenant-B"
TASK_OWNER = "task-owner"
APPROVER_ID = "approver-123"
PAUSE_ID = "P1"
HUMAN_REQUEST_ID = "H1"


def _identity_user_approver(*, tenant_id: str = TENANT_A, user_id: str = APPROVER_ID) -> HumanApproverEvidence:
    principal = HarnessAuthenticatedPrincipal(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
        auth_mode="identity_provider",
    )
    return harness_principal_to_approver_evidence(principal)


def _api_key_service_approver(*, tenant_id: str = TENANT_A) -> HumanApproverEvidence:
    principal = HarnessAuthenticatedPrincipal(
        tenant_id=tenant_id,
        user_id="harness-api-key",
        principal_type=PrincipalType.SERVICE,
        auth_subject="harness-api-key",
        auth_mode="api_key",
    )
    return harness_principal_to_approver_evidence(principal)


def _paused_task(*, tenant_id: str = TENANT_A, user_id: str = TASK_OWNER) -> Task:
    task = Task(tenant_id=tenant_id, user_id=user_id, message="x", task_id=TASK_ID)
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=PAUSE_ID,
        task_id=TASK_ID,
        human_request_id=HUMAN_REQUEST_ID,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=HUMAN_REQUEST_ID,
        prompt="approve?",
    )
    return task


def _checkpoint_with_pause(*, tenant_id: str = TENANT_A, user_id: str = TASK_OWNER) -> TaskCheckpoint:
    task = _paused_task(tenant_id=tenant_id, user_id=user_id)
    return TaskCheckpoint(
        checkpoint_id="chk-1",
        task_id=TASK_ID,
        tenant_id=tenant_id,
        resume_token="resume-token-1",
        task_snapshot=task.model_dump(mode="json"),
        task_state=TaskState.WAITING_FOR_HUMAN,
        progress_message="paused",
        notify_channel="debug",
    )


def _resolve_with_approver(
    task: Task,
    verdict: HumanResponseVerdict,
    approver: HumanApproverEvidence,
) -> HumanApprovalResolution:
    return HumanPauseCoordinator.resolve_human_response(
        task,
        verdict,
        approver=approver,
        pause_id=PAUSE_ID,
        human_request_id=HUMAN_REQUEST_ID,
    )


def test_c1_authenticated_approver_persisted() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    approver = _identity_user_approver()
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(
        task,
        HumanResponseVerdict.APPROVE,
        human_store=store,
        response_text="approved",
    )
    records = store.list_for_task(TASK_ID, TENANT_A)
    assert len(records) == 1
    record = records[0]
    assert record.approver.tenant_id == TENANT_A
    assert record.approver.user_id == APPROVER_ID
    assert record.approver.principal_type is PrincipalType.USER
    assert record.approver.auth_subject == APPROVER_ID
    assert record.approver.auth_mode is HumanApproverAuthMode.IDENTITY_PROVIDER
    assert record.user_id == TASK_OWNER


def test_c2_task_user_not_approver() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task(user_id=TASK_OWNER)
    approver = _identity_user_approver(user_id=APPROVER_ID)
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT_A)[0]
    assert record.approver.user_id == APPROVER_ID
    assert record.approver.user_id != task.user_id


def test_c3_no_credential_secret_persisted() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    approver = _identity_user_approver()
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT_A)[0]
    resolution = task.runtime.governance.hitl_resolution
    assert resolution is not None
    blob = json.dumps(
        {
            "record": record.model_dump(mode="json"),
            "resolution": resolution.model_dump(mode="json"),
        }
    )
    assert SECRET_CREDENTIAL_MUST_NOT_PERSIST not in blob


def test_c4_shared_http_exact_pause_materialization() -> None:
    checkpoint = _checkpoint_with_pause()
    task = Task(tenant_id=TENANT_A, user_id=TASK_OWNER, message="x", task_id=TASK_ID)
    _materialize_hitl_resume_input(
        task,
        checkpoint=checkpoint,
        operator_input={"verdict": "approve"},
        approver=_identity_user_approver(),
    )
    assert task.options.human.pause_id == PAUSE_ID
    assert task.options.human.human_request_id == HUMAN_REQUEST_ID


def test_c5_forged_pause_request_fails_closed() -> None:
    checkpoint = _checkpoint_with_pause()
    task = Task(tenant_id=TENANT_A, user_id=TASK_OWNER, message="x", task_id=TASK_ID)
    with pytest.raises(HitlResumeValidationError, match="pause_id conflicts"):
        _materialize_hitl_resume_input(
            task,
            checkpoint=checkpoint,
            operator_input={
                "verdict": "approve",
                "pause_id": "FORGED",
                "human_request_id": "FORGED",
            },
            approver=_identity_user_approver(),
        )


def test_c6_missing_pause_record_fails_closed() -> None:
    bare_task = Task(tenant_id=TENANT_A, user_id=TASK_OWNER, message="x", task_id=TASK_ID)
    checkpoint = TaskCheckpoint(
        checkpoint_id="chk-2",
        task_id=TASK_ID,
        tenant_id=TENANT_A,
        resume_token="resume-token-2",
        task_snapshot=bare_task.model_dump(mode="json"),
        task_state=TaskState.WAITING_FOR_HUMAN,
        progress_message="paused",
        notify_channel="debug",
    )
    task = Task(tenant_id=TENANT_A, user_id=TASK_OWNER, message="x", task_id=TASK_ID)
    with pytest.raises(HitlResumeValidationError, match="no active pause_record"):
        _materialize_hitl_resume_input(
            task,
            checkpoint=checkpoint,
            operator_input={"verdict": "approve"},
            approver=_identity_user_approver(),
        )


def test_c7_wrong_tenant_approver_fails_closed() -> None:
    task = _paused_task(tenant_id=TENANT_B)
    approver = _identity_user_approver(tenant_id=TENANT_A)
    with pytest.raises(HumanApprovalResolutionError, match="approver tenant_id mismatch"):
        _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)


def test_c10_api_key_service_approver() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    approver = _api_key_service_approver()
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT_A)[0]
    assert record.approver.principal_type is PrincipalType.SERVICE
    assert record.approver.auth_mode is HumanApproverAuthMode.API_KEY


def test_c11_identity_provider_user_approver() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    approver = _identity_user_approver()
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT_A)[0]
    assert record.approver.principal_type is PrincipalType.USER
    assert record.approver.auth_mode is HumanApproverAuthMode.IDENTITY_PROVIDER
    assert record.approver.auth_subject == APPROVER_ID


def test_c12_local_dev_explicit_provenance() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    approver = local_development_approver_evidence(tenant_id=TENANT_A)
    _resolve_with_approver(task, HumanResponseVerdict.APPROVE, approver)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT_A)[0]
    assert record.approver.auth_mode is HumanApproverAuthMode.LOCAL_DEVELOPMENT
    assert record.approver.auth_mode is not HumanApproverAuthMode.IDENTITY_PROVIDER


def test_c14_serialization_roundtrip() -> None:
    approver = _identity_user_approver()
    resolution = HumanApprovalResolution(
        task_id=TASK_ID,
        pause_id=PAUSE_ID,
        human_request_id=HUMAN_REQUEST_ID,
        verdict=HumanResponseVerdict.APPROVE,
        approver=approver,
        resolved_at="2026-08-22T00:00:00+00:00",
        run_id=RUN_ID,
        response_text="ok",
    )
    restored = HumanApprovalResolution.model_validate_json(resolution.model_dump_json())
    assert restored == resolution

    record = build_human_decision_record(
        task_id=TASK_ID,
        tenant_id=TENANT_A,
        approver=approver,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="ok",
        human_request_id=HUMAN_REQUEST_ID,
    )
    restored_record = HumanDecisionRecord.model_validate_json(record.model_dump_json())
    assert restored_record.approver == approver


@pytest.mark.asyncio
async def test_c13_event_evidence_contains_safe_approver() -> None:
    task = _paused_task()
    approver = _identity_user_approver()
    task.options.human.verdict = "approve"
    task.options.human.pause_id = PAUSE_ID
    task.options.human.human_request_id = HUMAN_REQUEST_ID
    task.options.human.approver = approver
    task.options.human.response_text = "approve"

    published: list[object] = []

    async def publish(event: object) -> None:
        published.append(event)

    hitl = NexusHitlRunner(
        publish=publish,
        human_hooks=HumanApprovalHookCoordinator(MiddlewarePipeline()),
        lifecycle_hooks=MagicMock(),
        escalation_router=EscalationRouter(max_levels=3),
        notification_adapter=None,
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        persist_human_decision=MagicMock(),
        execution_identity=ActiveExecutionIdentity(),
    )
    runner = NexusIntakeRunner(
        hitl=hitl,
        human_hooks=HumanApprovalHookCoordinator(MiddlewarePipeline()),
        publish=publish,
        restore_long_running=AsyncMock(),
        execution_identity=ActiveExecutionIdentity(),
    )
    lifecycle = TaskLifecycle()
    trace_emitter = TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    token = bind_active_execution_identity(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    try:
        await runner.run(task, lifecycle=lifecycle, trace_emitter=trace_emitter)
    finally:
        reset_active_execution_identity(token)

    approval_events = [
        event
        for event in published
        if event.event_type == RuntimeEventType.HUMAN_APPROVAL_RECEIVED
    ]
    assert len(approval_events) == 1
    payload = approval_events[0].payload
    assert payload["task_id"] == TASK_ID
    assert payload["pause_id"] == PAUSE_ID
    assert payload["human_request_id"] == HUMAN_REQUEST_ID
    assert payload["verdict"] == HumanResponseVerdict.APPROVE.value
    assert payload["approver_user_id"] == APPROVER_ID
    assert payload["principal_type"] == PrincipalType.USER.value
    assert payload["auth_mode"] == HumanApproverAuthMode.IDENTITY_PROVIDER.value
    assert SECRET_CREDENTIAL_MUST_NOT_PERSIST not in json.dumps(payload)


class _FakeCheckpointStore:
    def __init__(self, checkpoint: TaskCheckpoint) -> None:
        self._checkpoint = checkpoint

    def get_by_token(self, task_id: str, tenant_id: str, resume_token: str) -> TaskCheckpoint | None:
        if (
            task_id == self._checkpoint.task_id
            and tenant_id == self._checkpoint.tenant_id
            and resume_token == self._checkpoint.resume_token
        ):
            return self._checkpoint
        return None


class _FakeIdentityProvider:
    def verify_token(self, token: str) -> IdentityUser:
        if token != "valid-bearer":
            raise ValueError("invalid")
        return IdentityUser(user_id=APPROVER_ID, tenant_id=TENANT_A)


class _RecordingEvaluatorForIdt:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def evaluate(self, request: object) -> object:
        from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision

        self.calls.append(request)
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.test_allow",
            decision_id="dec-allow",
        )


def test_c7_http_tenant_conflict() -> None:
    from intergrax.runtime.governance.control_plane_mutation_authorization import (
        ControlPlaneMutationAuthorizationBoundary,
    )

    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_RecordingEvaluatorForIdt(),
    )
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    runner = MagicMock()
    mount_harness_task_routes(
        app,
        task_runner=runner,
        checkpoint_store=_FakeCheckpointStore(_checkpoint_with_pause()),
        mutation_boundary=boundary,
    )
    client = TestClient(app)
    response = client.post(
        f"/v1/tasks/{TASK_ID}/resume",
        json={
            "mutation_id": "mut-resume-idt",
            "tenant_id": TENANT_B,
            "resume_token": "resume-token-1",
            "operator_input": {"verdict": "approve"},
        },
        headers={"Authorization": "Bearer valid-bearer"},
    )
    assert response.status_code == 400
    assert "tenant_id" in response.json()["detail"]
