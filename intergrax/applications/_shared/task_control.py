# © Artur Czarnecki. All rights reserved.

"""Harness task lifecycle control — cancel, resume, autonomy (FLOW-CTL, REL-ADV.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import build_checkpoint_resume_task
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


@dataclass(frozen=True, slots=True)
class TaskControlResult:
    task_id: str
    action: str
    accepted: bool
    detail: str = ""
    state: str | None = None
    metadata: dict[str, Any] | None = None


async def cancel_active_task(task_id: str, *, reason: str = "operator_cancel") -> TaskControlResult:
    task = await ActiveTaskRegistry.get(task_id)
    if task is None:
        return TaskControlResult(
            task_id=task_id,
            action="cancel",
            accepted=False,
            detail="task_not_active",
        )
    CancellationCoordinator.request(task, reason=reason)
    return TaskControlResult(
        task_id=task_id,
        action="cancel",
        accepted=True,
        detail=reason,
        state=task.state.value,
    )


async def set_task_autonomy(task_id: str, level: AutonomyLevel) -> TaskControlResult:
    task = await ActiveTaskRegistry.get(task_id)
    if task is None:
        return TaskControlResult(
            task_id=task_id,
            action="set_autonomy",
            accepted=False,
            detail="task_not_active",
        )
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


async def resume_task_with_token(
    runner: UnifiedTaskRunner,
    *,
    task_id: str,
    resume_token: str,
    operator_input: dict[str, Any] | None = None,
    checkpoint: TaskCheckpoint,
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
    return await runner.run_task(task)
