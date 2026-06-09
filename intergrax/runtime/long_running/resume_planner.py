# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build resume Task payloads for scheduler-driven execution (§26, J.4)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)


def timeout_action_to_verdict(action: AgentDecisionType) -> HumanResponseVerdict:
    if action == AgentDecisionType.ESCALATE:
        return HumanResponseVerdict.ESCALATE
    if action in (AgentDecisionType.FAIL, AgentDecisionType.CANCEL):
        return HumanResponseVerdict.REJECT
    raise ValueError(f"unsupported default_on_timeout action: {action.value}")


def build_timeout_resume_task(
    checkpoint: TaskCheckpoint,
    *,
    verdict: HumanResponseVerdict,
    action: AgentDecisionType,
) -> Task:
    task = _base_resume_task(checkpoint)
    task.options.human = TaskHumanInput(
        response_text=f"scheduler:timeout:{action.value}",
        verdict=verdict.value,
    )
    _apply_verdict_metadata(task, verdict)
    task.metadata["scheduler_timeout"] = True
    task.metadata["scheduler_timeout_action"] = action.value
    task.sync_metadata()
    return task


def build_scheduled_resume_task(
    checkpoint: TaskCheckpoint,
    entry: ScheduledResume,
) -> Task:
    task = _base_resume_task(checkpoint)
    extra = dict(entry.resume_metadata or {})
    verdict_raw = extra.pop("verdict", None)
    if verdict_raw:
        verdict = HumanResponseVerdict(str(verdict_raw))
        task.options.human = TaskHumanInput(
            response_text=str(extra.pop("response_text", f"scheduler:delayed:{verdict.value}")),
            verdict=verdict.value,
        )
        _apply_verdict_metadata(task, verdict)
    elif extra.get("human_approved"):
        task.options.human = TaskHumanInput(
            response_text=str(extra.pop("response_text", "scheduler:delayed:approve")),
            verdict=HumanResponseVerdict.APPROVE.value,
        )
        task.metadata["human_approved"] = True
    task.metadata["scheduler_delayed_resume"] = True
    task.metadata["schedule_id"] = entry.schedule_id
    for key, value in extra.items():
        task.metadata[key] = value
    task.sync_metadata()
    return task


def build_checkpoint_resume_task(checkpoint: TaskCheckpoint) -> Task:
    """Public helper for operator/API resume (FLOW-CTL.4)."""
    return _base_resume_task(checkpoint)


def _base_resume_task(checkpoint: TaskCheckpoint) -> Task:
    task = Task.model_validate(checkpoint.task_snapshot)
    task.options.long_running = TaskLongRunningOptions(
        enabled=True,
        notify_channel=task.options.long_running.notify_channel,
        checkpoint_on_pause=task.options.long_running.checkpoint_on_pause,
        resume_token=checkpoint.resume_token,
    )
    task.options = TaskExecutionOptions.model_validate(task.options.model_dump())
    task.metadata["resume_token"] = checkpoint.resume_token
    return task


def _apply_verdict_metadata(task: Task, verdict: HumanResponseVerdict) -> None:
    if verdict == HumanResponseVerdict.APPROVE:
        task.metadata["human_approved"] = True
    elif verdict == HumanResponseVerdict.REJECT:
        task.metadata["human_rejected"] = True
    elif verdict == HumanResponseVerdict.ESCALATE:
        task.metadata["human_escalated"] = True
