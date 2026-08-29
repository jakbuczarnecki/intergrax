# © Artur Czarnecki. All rights reserved.

"""IDT-FIX-D — canonical RunId/AttemptId on HITL/lifecycle provenance paths."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.human.escalation import EscalationRouter
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator, human_approval_hook_context
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
ATTEMPT_ID = mint_attempt_id()
PAUSE_ID = "pause-distinct"
HUMAN_REQUEST_ID = "hr-distinct"
TENANT = "tenant-hitl"
APPROVER = local_development_approver_evidence(tenant_id=TENANT)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_IDT_D_PRODUCTION_FILES = (
    "intergrax/runtime/nexus/orchestration/human_response.py",
    "intergrax/runtime/nexus/orchestration/intake_runner.py",
    "intergrax/runtime/nexus/orchestration/hitl_runner.py",
    "intergrax/runtime/human/hitl_hooks.py",
    "intergrax/runtime/hooks/nexus_lifecycle_hooks.py",
    "intergrax/runtime/nexus/nexus_loop.py",
)
_FORBIDDEN_RUN_ID_PATTERNS = (
    re.compile(r"run_id\s*=\s*task\.task_id"),
    re.compile(r"run_id\s*=\s*task_id\b"),
    re.compile(r"""["']run_id["']\s*:\s*task\.task_id"""),
)


def _paused_task() -> Task:
    task = Task(tenant_id=TENANT, user_id="task-owner", message="x", task_id=TASK_ID)
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


def _resolve(
    task: Task,
    verdict: HumanResponseVerdict,
    *,
    run_id: str | None = None,
) -> HumanApprovalResolution:
    return HumanPauseCoordinator.resolve_human_response(
        task,
        verdict,
        approver=APPROVER,
        pause_id=PAUSE_ID,
        human_request_id=HUMAN_REQUEST_ID,
        run_id=run_id,
        response_text=task.options.human.response_text,
    )


def _build_intake_runner(
    *,
    human_store: InMemoryHumanDecisionPersistence | None = None,
) -> tuple[NexusIntakeRunner, list[object], NexusHitlRunner]:
    published: list[object] = []
    human_hooks = HumanApprovalHookCoordinator(MiddlewarePipeline())
    execution_identity = ActiveExecutionIdentity()

    async def publish(event: object, **kwargs: object) -> None:
        published.append(event)

    async def finish_task(task: Task, *args: object, **kwargs: object) -> TaskResult:
        return TaskResult(task_id=task.task_id, state=task.state)

    def persist_decision(
        task: Task,
        verdict: HumanResponseVerdict,
        *,
        response_text: str = "",
    ) -> None:
        persist_human_decision(
            task,
            verdict,
            human_store=human_store,
            response_text=response_text,
        )

    hitl = NexusHitlRunner(
        publish=publish,
        human_hooks=human_hooks,
        lifecycle_hooks=MagicMock(),
        escalation_router=EscalationRouter(max_levels=3),
        notification_adapter=None,
        finish_task=finish_task,
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        persist_human_decision=persist_decision,
        execution_identity=execution_identity,
    )
    runner = NexusIntakeRunner(
        hitl=hitl,
        human_hooks=human_hooks,
        publish=publish,
        restore_long_running=AsyncMock(),
        execution_identity=execution_identity,
    )
    return runner, published, hitl


def _set_human_response(task: Task, *, verdict: HumanResponseVerdict, response_text: str) -> None:
    task.options.human.response_text = response_text
    task.options.human.verdict = verdict.value
    task.options.human.pause_id = PAUSE_ID
    task.options.human.human_request_id = HUMAN_REQUEST_ID
    task.options.human.approver = APPROVER
    task.sync_metadata()


def test_d1_distinct_identity_human_decision_persistence() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    _resolve(task, HumanResponseVerdict.APPROVE, run_id=RUN_ID)
    persist_human_decision(
        task,
        HumanResponseVerdict.APPROVE,
        human_store=store,
        response_text="approved",
        run_id=RUN_ID,
    )
    record = store.list_for_task(TASK_ID, TENANT)[0]
    assert record.task_id == TASK_ID
    assert record.run_id == RUN_ID
    assert record.run_id != TASK_ID


def test_d2_approve_resolution_run_id() -> None:
    task = _paused_task()
    resolution = _resolve(task, HumanResponseVerdict.APPROVE, run_id=RUN_ID)
    assert resolution.run_id == RUN_ID
    assert resolution.run_id != TASK_ID


@pytest.mark.asyncio
async def test_d3_reject_resolution_and_event_run_id() -> None:
    task = _paused_task()
    _set_human_response(task, verdict=HumanResponseVerdict.REJECT, response_text="reject")
    store = InMemoryHumanDecisionPersistence()
    runner, published, _ = _build_intake_runner(human_store=store)
    token = bind_active_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=mint_execution_id(),
    )
    try:
        await runner.run(
            task,
            lifecycle=TaskLifecycle(),
            trace_emitter=TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID),
        )
    finally:
        reset_active_execution_identity(token)

    resolution = task.runtime.governance.hitl_resolution
    assert resolution is not None
    assert resolution.run_id == RUN_ID
    assert resolution.run_id != TASK_ID

    record = store.list_for_task(TASK_ID, TENANT)[0]
    assert record.run_id == RUN_ID

    events = [
        event
        for event in published
        if event.event_type == RuntimeEventType.HUMAN_APPROVAL_RECEIVED
    ]
    assert len(events) == 1
    assert events[0].run_id == RUN_ID
    assert events[0].attempt_id == ATTEMPT_ID


@pytest.mark.asyncio
async def test_d4_escalate_resolution_and_event_run_id() -> None:
    task = _paused_task()
    _set_human_response(task, verdict=HumanResponseVerdict.ESCALATE, response_text="escalate")
    store = InMemoryHumanDecisionPersistence()
    runner, published, _ = _build_intake_runner(human_store=store)
    token = bind_active_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=mint_execution_id(),
    )
    try:
        await runner.run(
            task,
            lifecycle=TaskLifecycle(),
            trace_emitter=TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID),
        )
    finally:
        reset_active_execution_identity(token)

    resolution = task.runtime.governance.hitl_resolution
    assert resolution is not None
    assert resolution.run_id == RUN_ID

    record = store.list_for_task(TASK_ID, TENANT)[0]
    assert record.run_id == RUN_ID

    events = [
        event
        for event in published
        if event.event_type == RuntimeEventType.INTERRUPT_ESCALATED
    ]
    assert len(events) == 1
    assert events[0].run_id == RUN_ID
    assert events[0].attempt_id == ATTEMPT_ID


@pytest.mark.asyncio
async def test_d5_human_approval_received_three_way_identity() -> None:
    task = _paused_task()
    _set_human_response(task, verdict=HumanResponseVerdict.APPROVE, response_text="approve")

    runner, published, _ = _build_intake_runner()
    token = bind_active_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=mint_execution_id(),
    )
    try:
        await runner.run(
            task,
            lifecycle=TaskLifecycle(),
            trace_emitter=TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID),
        )
    finally:
        reset_active_execution_identity(token)

    events = [
        event
        for event in published
        if event.event_type == RuntimeEventType.HUMAN_APPROVAL_RECEIVED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.task_id == TASK_ID
    assert event.run_id == RUN_ID
    assert event.attempt_id == ATTEMPT_ID
    assert event.task_id != event.run_id
    assert event.run_id != event.attempt_id


@pytest.mark.asyncio
async def test_d6_checkpoint_resume_preserves_execution_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    paused = _paused_task()
    checkpoint = TaskCheckpoint(
        checkpoint_id="chk-idt-d",
        task_id=TASK_ID,
        tenant_id=TENANT,
        resume_token="resume-token-idt-d",
        task_snapshot=paused.model_dump(mode="json"),
        task_state=TaskState.WAITING_FOR_HUMAN,
        progress_message="paused",
        notify_channel="debug",
        runtime=minimal_runtime_checkpoint(
            task_id=TASK_ID,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        ),
    )

    captured: dict[str, object] = {}

    async def _fake_handle_task(
        _self: object,
        task: Task,
        *,
        run_id: str,
        attempt_id: str | None = None,
    ) -> TaskResult:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        captured["task_id"] = task.task_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    class _FakeCheckpointStore:
        def get_by_token(self, task_id: str, tenant_id: str, resume_token: str) -> TaskCheckpoint | None:
            if task_id == TASK_ID and tenant_id == TENANT and resume_token == checkpoint.resume_token:
                return checkpoint
            return None

    service = DebugHitlResumeService(
        AgentRegistry(),
        checkpoint_store=_FakeCheckpointStore(),
    )
    monkeypatch.setattr(
        "intergrax.debug.hitl_service.NexusLoop.handle_task",
        _fake_handle_task,
    )

    await service.resume_with_human_response(
        TASK_ID,
        TENANT,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="approve",
        resume_token=checkpoint.resume_token,
        approver=APPROVER,
    )

    assert captured["task_id"] == TASK_ID
    assert captured["run_id"] == RUN_ID
    assert captured["attempt_id"] == ATTEMPT_ID
    assert captured["run_id"] != captured["task_id"]


def test_d7_no_taskid_as_runid_static_regression() -> None:
    violations: list[str] = []
    for rel_path in _IDT_D_PRODUCTION_FILES:
        source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_RUN_ID_PATTERNS:
            if pattern.search(source):
                violations.append(f"{rel_path}: {pattern.pattern}")
    assert violations == []


def test_d8_missing_run_id_persists_null_not_task_id() -> None:
    store = InMemoryHumanDecisionPersistence()
    task = _paused_task()
    _resolve(task, HumanResponseVerdict.APPROVE, run_id=None)
    persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    record = store.list_for_task(TASK_ID, TENANT)[0]
    assert record.run_id is None
    assert record.run_id != TASK_ID


def test_d8b_human_approval_hook_context_uses_active_run_id() -> None:
    task = _paused_task()
    token = bind_active_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=mint_execution_id(),
    )
    try:
        ctx = human_approval_hook_context(task, verdict=HumanResponseVerdict.APPROVE.value)
    finally:
        reset_active_execution_identity(token)
    assert ctx.task_id == TASK_ID
    assert ctx.run_id == RUN_ID
    assert ctx.run_id != TASK_ID


def test_d12_serialization_preserves_distinct_identities() -> None:
    resolution = HumanApprovalResolution(
        task_id=TASK_ID,
        pause_id=PAUSE_ID,
        human_request_id=HUMAN_REQUEST_ID,
        verdict=HumanResponseVerdict.APPROVE,
        approver=APPROVER,
        resolved_at="2026-08-22T00:00:00+00:00",
        run_id=RUN_ID,
        response_text="ok",
    )
    restored = HumanApprovalResolution.model_validate_json(resolution.model_dump_json())
    assert restored.task_id == TASK_ID
    assert restored.run_id == RUN_ID
    assert restored.task_id != restored.run_id
