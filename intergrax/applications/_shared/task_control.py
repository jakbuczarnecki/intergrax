# © Artur Czarnecki. All rights reserved.

"""Harness task lifecycle control — cancel, resume, autonomy (FLOW-CTL, REL-ADV.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.contracts.human_approver import HumanApproverEvidence, local_development_approver_evidence
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import build_checkpoint_resume_task
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class HitlResumeValidationError(ValueError):
    """Fail-closed validation for shared HITL resume surfaces."""


@dataclass(frozen=True, slots=True)
class TaskControlResult:
    task_id: str
    action: str
    accepted: bool
    detail: str = ""
    state: str | None = None
    metadata: dict[str, Any] | None = None


async def cancel_active_task(task_id: str, *, reason: str = "operator_cancel") -> TaskControlResult:
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="task_not_active",
        )
    task = binding.task
    CancellationCoordinator.request(task, reason=reason)
    return TaskControlResult(
        task_id=task_id,
        action="cancel",
        accepted=True,
        detail=reason,
        state=task.state.value,
    )


async def set_task_autonomy(task_id: str, level: AutonomyLevel) -> TaskControlResult:
    binding = await ActiveTaskRegistry.get(task_id)
    if binding is None:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="task_not_active",
        )
    task = binding.task
    previous = task.options.governance.autonomy_level
    task.options.governance.autonomy_level = level
    task.metadata["autonomy_level"] = level.value
    task.metadata["autonomy_level_previous"] = previous.value if previous else None
    task.metadata["autonomy_level_changed"] = True
    task.sync_metadata()
    return TaskControlResult(
        task_id=task_id,
        action="set_autonomy",
        accepted=True,
        detail=level.value,
        state=task.state.value,
        metadata={"previous": previous.value if previous else None},
    )


def _pause_record_from_checkpoint(checkpoint: TaskCheckpoint) -> TaskPauseRecord | None:
    snapshot = Task.model_validate(checkpoint.task_snapshot)
    return snapshot.runtime.governance.pause_record


def _materialize_hitl_resume_input(
    task: Task,
    *,
    checkpoint: TaskCheckpoint,
    operator_input: dict[str, Any] | None,
    approver: HumanApproverEvidence | None,
) -> None:
    verdict = (operator_input or {}).get("verdict")
    if not verdict:
        return

    pause_record = _pause_record_from_checkpoint(checkpoint)
    if pause_record is None:
        raise HitlResumeValidationError(
            "checkpoint has no active pause_record for human approval resume"
        )

    forged_pause_id = (operator_input or {}).get("pause_id")
    if forged_pause_id is not None and forged_pause_id != pause_record.pause_id:
        raise HitlResumeValidationError(
            "operator_input pause_id conflicts with checkpoint pause_record"
        )

    forged_request_id = (operator_input or {}).get("human_request_id")
    if forged_request_id is not None and forged_request_id != pause_record.human_request_id:
        raise HitlResumeValidationError(
            "operator_input human_request_id conflicts with checkpoint pause_record"
        )

    task.options.human.pause_id = pause_record.pause_id
    task.options.human.human_request_id = pause_record.human_request_id

    if approver is not None:
        task.options.human.approver = approver
    else:
        task.options.human.approver = local_development_approver_evidence(
            tenant_id=task.tenant_id,
        )


async def resume_task_with_token(
    runner: UnifiedTaskRunner,
    *,
    task_id: str,
    resume_token: str,
    operator_input: dict[str, Any] | None = None,
    checkpoint: TaskCheckpoint,
    approver: HumanApproverEvidence | None = None,
) -> TaskResult:
    task = build_checkpoint_resume_task(checkpoint)
    task.task_id = task_id
    task.options.long_running.resume_token = resume_token
    if operator_input:
        verdict = operator_input.get("verdict")
        if verdict:
            task.options.human.verdict = str(verdict)
        response_text = operator_input.get("response_text")
        if response_text:
            task.options.human.response_text = str(response_text)
    _materialize_hitl_resume_input(
        task,
        checkpoint=checkpoint,
        operator_input=operator_input,
        approver=approver,
    )
    return await runner.run_task(task, resume_checkpoint=checkpoint)
