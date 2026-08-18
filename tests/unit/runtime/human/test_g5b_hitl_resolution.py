# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
    DeclarativeHitlGrantError,
)
from intergrax.runtime.human.hitl_hooks import HumanApprovalHookCoordinator
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanApprovalResolutionError, HumanPauseCoordinator
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
ATTEMPT_ID = mint_attempt_id()


def _active_pause(
    task: Task,
    *,
    pause_id: str = "pause-1",
    human_request_id: str = "hr-1",
) -> None:
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=pause_id,
        task_id=task.task_id,
        human_request_id=human_request_id,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=human_request_id,
        prompt="approve?",
    )


def _resolve(
    task: Task,
    verdict: HumanResponseVerdict,
    *,
    pause_id: str = "pause-1",
    human_request_id: str = "hr-1",
    **kwargs: object,
) -> HumanApprovalResolution:
    return HumanPauseCoordinator.resolve_human_response(
        task,
        verdict,
        pause_id=pause_id,
        human_request_id=human_request_id,
        **kwargs,
    )


def _apply_pause(task: Task, *, human_request_id: str) -> TaskPauseRecord:
    execution = AgentExecutionResult(
        agent_id="agent-1",
        run_id=RUN_ID,
        status=AgentExecutionStatus.NEEDS_INPUT,
        human_request=HumanRequest(request_id=human_request_id, prompt="approve?"),
    )
    HumanPauseCoordinator.apply_pause(task, execution)
    assert task.runtime.governance.pause_record is not None
    return task.runtime.governance.pause_record


def _pending(
    task: Task,
    *,
    pause_id: str = "pause-1",
    human_request_id: str = "hr-1",
) -> DeclarativeHitlPendingApproval:
    return DeclarativeHitlPendingApproval(
        invocation_scope_id="dhr_scope",
        task_id=task.task_id,
        run_id=RUN_ID,
        step_id="step-1",
        tool_id="tool.a",
        idempotency_key="idem-1",
        matched_rule_ids=("rule-1",),
        human_request_id=human_request_id,
        policy_provenance_digest="digest-1",
        agent_id="agent-1",
        pause_id=pause_id,
        created_at="2026-08-14T00:00:00+00:00",
    )


def test_valid_active_pause_approve_persists_canonical_resolution() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    resolution = _resolve(
        task,
        HumanResponseVerdict.APPROVE,
        run_id=RUN_ID,
        response_text="approve",
    )
    assert resolution.verdict is HumanResponseVerdict.APPROVE
    assert resolution.task_id == TASK_ID
    assert resolution.pause_id == "pause-1"
    assert resolution.human_request_id == "hr-1"
    assert resolution.run_id == RUN_ID
    assert task.runtime.governance.hitl_resolution == resolution


def test_resolution_is_immutable() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    resolution = _resolve(task, HumanResponseVerdict.APPROVE)
    with pytest.raises(Exception):
        resolution.verdict = HumanResponseVerdict.REJECT


def test_approve_without_active_pause_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    with pytest.raises(HumanApprovalResolutionError, match="no active pause record"):
        _resolve(task, HumanResponseVerdict.APPROVE)


def test_approve_missing_pause_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="pause_id required"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            human_request_id="hr-1",
        )


def test_approve_missing_human_request_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="human_request_id required"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            pause_id="pause-1",
        )


def test_reject_missing_pause_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="pause_id required"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.REJECT,
            human_request_id="hr-1",
        )


def test_reject_missing_human_request_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="human_request_id required"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.REJECT,
            pause_id="pause-1",
        )


def test_approve_wrong_pause_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
        _resolve(task, HumanResponseVerdict.APPROVE, pause_id="pause-stale")


def test_approve_wrong_human_request_id_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="human_request_id mismatch"):
        _resolve(task, HumanResponseVerdict.APPROVE, human_request_id="hr-stale")


def test_approve_already_resolved_pause_cannot_authorize_again() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    _resolve(task, HumanResponseVerdict.APPROVE)
    with pytest.raises(HumanApprovalResolutionError, match="already resolved"):
        _resolve(task, HumanResponseVerdict.APPROVE)


def test_reject_wrong_pause_request_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
        _resolve(task, HumanResponseVerdict.REJECT, pause_id="pause-stale")


def test_stale_response_against_new_pause_fails_closed() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, pause_id="pause-A", human_request_id="hr-A")
    _resolve(task, HumanResponseVerdict.APPROVE, pause_id="pause-A", human_request_id="hr-A")

    pause_b = _apply_pause(task, human_request_id="hr-B")
    assert pause_b.pause_id != "pause-A"
    assert task.runtime.governance.hitl_resolution is None

    with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
        _resolve(task, HumanResponseVerdict.APPROVE, pause_id="pause-A", human_request_id="hr-A")

    assert task.runtime.governance.pause_record == pause_b
    assert task.runtime.governance.paused is True
    assert task.runtime.governance.hitl_resolution is None
    assert task.runtime.governance.declarative_hitl_grant is None


def test_two_independent_approval_cycles() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, pause_id="pause-A", human_request_id="hr-A")
    resolution_a = _resolve(
        task,
        HumanResponseVerdict.APPROVE,
        pause_id="pause-A",
        human_request_id="hr-A",
    )
    assert resolution_a.pause_id == "pause-A"

    pause_b = _apply_pause(task, human_request_id="hr-B")
    resolution_b = _resolve(
        task,
        HumanResponseVerdict.APPROVE,
        pause_id=pause_b.pause_id,
        human_request_id="hr-B",
    )
    assert resolution_b.pause_id == pause_b.pause_id
    assert resolution_b.human_request_id == "hr-B"
    assert resolution_b.verdict is HumanResponseVerdict.APPROVE
    assert task.runtime.governance.hitl_resolution == resolution_b


@pytest.mark.asyncio
async def test_intake_runner_passes_explicit_pause_identity() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, pause_id="pause-intake", human_request_id="hr-intake")
    task.options.human.verdict = "approve"
    task.options.human.pause_id = "pause-intake"
    task.options.human.human_request_id = "hr-intake"
    task.options.human.response_text = "approve"

    runner = NexusIntakeRunner(
        hitl=MagicMock(),
        human_hooks=HumanApprovalHookCoordinator(MiddlewarePipeline()),
        publish=AsyncMock(),
        restore_long_running=AsyncMock(),
        execution_identity=ActiveExecutionIdentity(),
    )
    lifecycle = TaskLifecycle()
    trace_emitter = TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    token = bind_active_execution_identity(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    try:
        outcome = await runner.run(task, lifecycle=lifecycle, trace_emitter=trace_emitter)
    finally:
        reset_active_execution_identity(token)

    assert outcome.early_result is None
    resolution = task.runtime.governance.hitl_resolution
    assert resolution is not None
    assert resolution.pause_id == "pause-intake"
    assert resolution.human_request_id == "hr-intake"
    assert resolution.verdict is HumanResponseVerdict.APPROVE
    assert task.runtime.governance.paused is False
    assert task.options.human.pause_id is None
    assert task.options.human.human_request_id is None
    assert task.options.human.verdict is None


def test_declarative_pending_valid_approve_creates_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    task.runtime.governance.declarative_hitl_pending = _pending(task)
    _resolve(task, HumanResponseVerdict.APPROVE)
    grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert grant is not None
    assert grant.invocation_scope_id == "dhr_scope"
    assert grant.run_id == RUN_ID
    assert grant.step_id == "step-1"
    assert grant.tool_id == "tool.a"
    assert grant.idempotency_key == "idem-1"
    assert grant.matched_rule_ids == ("rule-1",)
    assert grant.policy_provenance_digest == "digest-1"
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is grant


def test_declarative_pending_pause_id_mismatch_no_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, pause_id="pause-1")
    task.runtime.governance.declarative_hitl_pending = _pending(task, pause_id="pause-other")
    _resolve(task, HumanResponseVerdict.APPROVE)
    with pytest.raises(DeclarativeHitlGrantError, match="pause_id mismatch"):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert task.runtime.governance.declarative_hitl_grant is None


def test_declarative_pending_human_request_id_mismatch_no_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, human_request_id="hr-1")
    task.runtime.governance.declarative_hitl_pending = _pending(
        task,
        human_request_id="hr-other",
    )
    _resolve(task, HumanResponseVerdict.APPROVE)
    with pytest.raises(DeclarativeHitlGrantError, match="human_request_id mismatch"):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert task.runtime.governance.declarative_hitl_grant is None


def test_declarative_stale_response_cannot_create_grant_for_new_pending() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task, pause_id="pause-A", human_request_id="hr-A")
    _resolve(task, HumanResponseVerdict.APPROVE, pause_id="pause-A", human_request_id="hr-A")

    pause_b = _apply_pause(task, human_request_id="hr-B")
    task.runtime.governance.declarative_hitl_pending = _pending(
        task,
        pause_id=pause_b.pause_id,
        human_request_id="hr-B",
    )

    with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
        _resolve(task, HumanResponseVerdict.APPROVE, pause_id="pause-A", human_request_id="hr-A")

    with pytest.raises(DeclarativeHitlGrantError, match="canonical approval resolution required"):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    assert task.runtime.governance.declarative_hitl_grant is None


def test_valid_reject_persists_resolution_clears_pending_no_grant() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    _active_pause(task)
    task.runtime.governance.declarative_hitl_pending = _pending(task)
    HumanPauseCoordinator.record_human_response(task, "reject")
    resolution = _resolve(
        task,
        HumanResponseVerdict.REJECT,
        response_text="reject",
    )
    assert resolution.verdict is HumanResponseVerdict.REJECT
    DeclarativeHitlGrantCoordinator.clear_pending_and_grant(task)
    assert task.runtime.governance.declarative_hitl_pending is None
    assert task.runtime.governance.declarative_hitl_grant is None
    task.runtime.governance.declarative_hitl_pending = _pending(task)
    with pytest.raises(DeclarativeHitlGrantError, match="not approve"):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
