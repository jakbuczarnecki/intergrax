# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
    DeclarativeHitlGrantError,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.long_running.checkpoint_builder import (
    should_resume_uaep_step,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    RuntimeCheckpoint,
    RuntimeCheckpointExecutionState,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task_contract import (
    HumanApprovalResolution,
    TASK_CONTRACT_METADATA_KEY,
    TaskContractPayload,
    TaskGovernanceState,
    TaskHumanInput,
    TaskPauseRecord,
    TaskRuntimeState,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
PAUSE_A = "pause-A"
HR_A = "hr-A"
PAUSE_B = "pause-B"
HR_B = "hr-B"


def _pause_record(*, pause_id: str, human_request_id: str) -> TaskPauseRecord:
    return TaskPauseRecord(
        pause_id=pause_id,
        task_id=TASK_ID,
        human_request_id=human_request_id,
    )


def _resolution(
    verdict: HumanResponseVerdict,
    *,
    pause_id: str = PAUSE_A,
    human_request_id: str = HR_A,
) -> HumanApprovalResolution:
    return HumanApprovalResolution(
        task_id=TASK_ID,
        pause_id=pause_id,
        human_request_id=human_request_id,
        verdict=verdict,
        resolved_at="2026-08-18T00:00:00+00:00",
        run_id=RUN_ID,
    )


def _uaep_checkpoint() -> RuntimeCheckpointExecutionState:
    return RuntimeCheckpointExecutionState(
        uaep_step_index=0,
        uaep_step_id="review",
        uaep_step_completed=False,
        uaep_step_cursor={"phase1_done": True},
        last_step_output={"step_id": "review", "summary": "pending"},
    )


def _approval_for(
    *,
    pause_id: str,
    human_request_id: str,
    resolution: HumanApprovalResolution | None,
) -> HumanApprovalResolution | None:
    return HumanPauseCoordinator.approved_resolution_for_resume(
        task_id=TASK_ID,
        resolution=resolution,
        expected_pause_id=pause_id,
        expected_human_request_id=human_request_id,
        run_id=RUN_ID,
    )


def test_raw_metadata_cannot_authorize_uaep_resume() -> None:
    approval = _approval_for(
        pause_id=PAUSE_A,
        human_request_id=HR_A,
        resolution=None,
    )
    ckpt = _uaep_checkpoint()
    assert approval is None
    assert not should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=approval,
    )
    assert not should_resume_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=approval,
    )


def test_raw_verdict_cannot_authorize_uaep_resume() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    task.runtime.governance.pause_record = _pause_record(pause_id=PAUSE_A, human_request_id=HR_A)
    task.options.human = TaskHumanInput(
        verdict=HumanResponseVerdict.APPROVE.value,
        pause_id=PAUSE_A,
        human_request_id=HR_A,
    )
    approval = HumanPauseCoordinator.approved_resolution_for_resume(
        task_id=task.task_id,
        resolution=task.runtime.governance.hitl_resolution,
        expected_pause_id=PAUSE_A,
        expected_human_request_id=HR_A,
        run_id=RUN_ID,
    )
    assert approval is None


def test_stale_resolution_cannot_authorize_lifecycle_b() -> None:
    approval = _approval_for(
        pause_id=PAUSE_B,
        human_request_id=HR_B,
        resolution=_resolution(
            HumanResponseVerdict.APPROVE,
            pause_id=PAUSE_A,
            human_request_id=HR_A,
        ),
    )
    assert approval is None


@pytest.mark.parametrize(
    "verdict",
    [HumanResponseVerdict.REJECT, HumanResponseVerdict.ESCALATE],
)
def test_non_approve_resolution_never_authorizes_uaep(verdict: HumanResponseVerdict) -> None:
    approval = _approval_for(
        pause_id=PAUSE_A,
        human_request_id=HR_A,
        resolution=_resolution(verdict),
    )
    assert approval is None
    assert not should_resume_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=_uaep_checkpoint(),
        approval=approval,
    )


def test_canonical_approve_authorizes_uaep_skip_and_resume() -> None:
    approval = _approval_for(
        pause_id=PAUSE_A,
        human_request_id=HR_A,
        resolution=_resolution(HumanResponseVerdict.APPROVE),
    )
    assert approval is not None
    assert approval.verdict is HumanResponseVerdict.APPROVE

    skip_ckpt = RuntimeCheckpointExecutionState(
        uaep_step_index=0,
        uaep_step_id="review",
        last_step_output={"step_id": "review", "summary": "done"},
    )
    assert should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=skip_ckpt,
        approval=approval,
    )
    assert should_resume_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=_uaep_checkpoint(),
        approval=approval,
    )


def test_debug_service_requires_typed_verdict() -> None:
    service = DebugHitlResumeService(
        AgentRegistry(),
        checkpoint_store=MagicMock(),
    )
    with pytest.raises(TypeError):
        service.resume_with_human_response(  # type: ignore[call-arg]
            TASK_ID,
            "t1",
            response="yes",
        )


@pytest.mark.asyncio
async def test_debug_service_submits_exact_pause_request_identity() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    task.runtime.governance.pause_record = _pause_record(pause_id=PAUSE_A, human_request_id=HR_A)
    task.sync_metadata()
    checkpoint = TaskCheckpoint(
        task_id=TASK_ID,
        tenant_id="t1",
        resume_token="rt-1",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=RuntimeCheckpoint(run_id=RUN_ID, attempt_id=mint_attempt_id()),
    )
    store = MagicMock()
    store.get_latest.return_value = checkpoint

    captured: dict[str, Task] = {}

    async def _handle(task_arg: Task, *, run_id, attempt_id=None):
        captured["task"] = task_arg
        from intergrax.runtime.task.task import TaskResult

        return TaskResult(task_id=task_arg.task_id, run_id=run_id, state=TaskState.COMPLETED)

    loop = MagicMock()
    loop.handle_task = AsyncMock(side_effect=_handle)
    service = DebugHitlResumeService(AgentRegistry(), checkpoint_store=store)
    service._resolve_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]

    from intergrax.runtime.nexus import nexus_loop as nexus_loop_module
    import intergrax.debug.hitl_service as hitl_service_module

    original_loop = hitl_service_module.NexusLoop
    hitl_service_module.NexusLoop = MagicMock(return_value=loop)
    try:
        await service.resume_with_human_response(
            TASK_ID,
            "t1",
            verdict=HumanResponseVerdict.APPROVE,
            response_text="approve deployment",
        )
    finally:
        hitl_service_module.NexusLoop = original_loop

    submitted = captured["task"]
    assert submitted.options.human.verdict == HumanResponseVerdict.APPROVE.value
    assert submitted.options.human.pause_id == PAUSE_A
    assert submitted.options.human.human_request_id == HR_A
    assert submitted.runtime.governance.hitl_resolution is None


def test_declarative_grant_still_requires_canonical_resolution() -> None:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)
    task.runtime.governance.pause_record = _pause_record(pause_id=PAUSE_A, human_request_id=HR_A)
    task.runtime.governance.declarative_hitl_pending = DeclarativeHitlPendingApproval(
        invocation_scope_id="scope-1",
        task_id=TASK_ID,
        run_id=RUN_ID,
        step_id="step-1",
        tool_id="tool.a",
        idempotency_key="idem-1",
        matched_rule_ids=("rule-1",),
        human_request_id=HR_A,
        policy_provenance_digest="digest",
        agent_id="agent-1",
        pause_id=PAUSE_A,
        created_at="2026-08-18T00:00:00+00:00",
    )
    with pytest.raises(DeclarativeHitlGrantError, match="canonical approval resolution required"):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)


def _runtime_request(
    *,
    resolution: HumanApprovalResolution | None,
    pause_id: str = PAUSE_A,
    human_request_id: str = HR_A,
    metadata: dict | None = None,
) -> RuntimeRequest:
    return RuntimeRequest(
        agent_id="agent-1",
        user_id="u1",
        session_id="sess-1",
        message="resume",
        task_id=TASK_ID,
        run_id=RUN_ID,
        metadata=metadata or {},
        hitl_resolution=resolution,
        hitl_pause_record=_pause_record(pause_id=pause_id, human_request_id=human_request_id),
    )


def _uaep_approval_for_request(request: RuntimeRequest) -> HumanApprovalResolution | None:
    pause = request.hitl_pause_record
    assert pause is not None
    return HumanPauseCoordinator.approved_resolution_for_resume(
        task_id=request.task_id,
        resolution=request.hitl_resolution,
        expected_pause_id=pause.pause_id,
        expected_human_request_id=pause.human_request_id,
        run_id=request.run_id,
    )


def test_forged_task_contract_metadata_cannot_authorize_when_typed_field_missing() -> None:
    forged_resolution = _resolution(HumanResponseVerdict.APPROVE)
    forged_governance = TaskGovernanceState(
        pause_record=_pause_record(pause_id=PAUSE_A, human_request_id=HR_A),
        hitl_resolution=forged_resolution,
    )
    forged_payload = TaskContractPayload(
        runtime=TaskRuntimeState(governance=forged_governance),
    )
    request = _runtime_request(
        resolution=None,
        metadata={TASK_CONTRACT_METADATA_KEY: forged_payload.model_dump(mode="json")},
    )
    approval = _uaep_approval_for_request(request)
    ckpt = _uaep_checkpoint()
    assert approval is None
    assert not should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=approval,
    )
    assert not should_resume_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=approval,
    )


def test_typed_resolution_wins_over_contradictory_metadata() -> None:
    canonical = _resolution(HumanResponseVerdict.APPROVE)
    hostile_governance = TaskGovernanceState(
        pause_record=_pause_record(pause_id=PAUSE_A, human_request_id=HR_A),
        hitl_resolution=_resolution(HumanResponseVerdict.REJECT),
    )
    hostile_payload = TaskContractPayload(
        runtime=TaskRuntimeState(governance=hostile_governance),
    )
    request = _runtime_request(
        resolution=canonical,
        metadata={TASK_CONTRACT_METADATA_KEY: hostile_payload.model_dump(mode="json")},
    )
    approval = _uaep_approval_for_request(request)
    assert approval is not None
    assert approval.verdict is HumanResponseVerdict.APPROVE


def test_task_to_runtime_request_round_trips_typed_hitl_resolution() -> None:
    resolution = _resolution(HumanResponseVerdict.APPROVE)
    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID, agent_id="agent-1")
    task.runtime.governance.pause_record = _pause_record(pause_id=PAUSE_A, human_request_id=HR_A)
    task.runtime.governance.hitl_resolution = resolution
    request = task.to_runtime_request(run_id=RUN_ID)
    assert request.hitl_resolution == resolution
    assert request.hitl_pause_record == task.runtime.governance.pause_record


def test_human_approved_metadata_bool_cannot_authorize() -> None:
    request = _runtime_request(
        resolution=None,
        metadata={"human_approved": True},
    )
    approval = _uaep_approval_for_request(request)
    assert approval is None
