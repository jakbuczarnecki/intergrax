# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build resume Task payloads for scheduler-driven execution (§26, J.4)."""

from __future__ import annotations


from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_approver import (
    HumanApproverAuthMode,
    HumanApproverEvidence,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.execution.execution_terminal.service import (
    ExecutionTerminalService,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    assert_checkpoint_resume_materialization_eligible,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduled_resume_metadata import (
    validate_scheduled_resume_metadata,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)


def _scheduler_approver_evidence(task: Task) -> HumanApproverEvidence:
    """Canonical approver for scheduler-driven HITL timeout / delayed resume."""
    return HumanApproverEvidence(
        tenant_id=task.tenant_id,
        user_id="long_running_scheduler",
        principal_type=PrincipalType.ORG_SYSTEM,
        auth_subject="long_running_scheduler",
        auth_mode=HumanApproverAuthMode.LOCAL_DEVELOPMENT,
    )


def _apply_scheduler_human_input(
    task: Task,
    *,
    verdict: HumanResponseVerdict,
    response_text: str,
) -> None:
    task.options.human = TaskHumanInput(
        response_text=response_text,
        verdict=verdict.value,
        approver=_scheduler_approver_evidence(task),
    )


def execution_identity_from_checkpoint(
    checkpoint: TaskCheckpoint,
) -> tuple[RunId, AttemptId]:
    runtime = checkpoint.runtime
    if runtime is None:
        raise ValueError(
            f"checkpoint {checkpoint.checkpoint_id!r} missing canonical execution identity"
        )
    return validate_run_id(runtime.run_id), validate_attempt_id(runtime.attempt_id)


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
    target_task_id: str,
    target_tenant_id: str,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
) -> Task:
    task = _base_resume_task(
        checkpoint,
        target_task_id=target_task_id,
        target_tenant_id=target_tenant_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
    )
    _apply_scheduler_human_input(
        task,
        verdict=verdict,
        response_text=f"scheduler:timeout:{action.value}",
    )
    task.metadata["scheduler_timeout"] = True
    task.metadata["scheduler_timeout_action"] = action.value
    task.sync_metadata()
    return task


def build_scheduled_resume_task(
    checkpoint: TaskCheckpoint,
    entry: ScheduledResume,
    *,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
) -> Task:
    """Build resume Task for delayed schedule — WHEN-only; no metadata-derived HITL."""
    task = _base_resume_task(
        checkpoint,
        target_task_id=entry.task_id,
        target_tenant_id=entry.tenant_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
    )
    validate_scheduled_resume_metadata(entry.resume_metadata)
    extra = dict(entry.resume_metadata or {})
    task.metadata["scheduler_delayed_resume"] = True
    task.metadata["schedule_id"] = entry.schedule_id
    for key, value in extra.items():
        task.metadata[key] = value
    task.sync_metadata()
    return task


def build_checkpoint_resume_task(
    checkpoint: TaskCheckpoint,
    *,
    target_task_id: str,
    target_tenant_id: str,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
) -> Task:
    """Operator/API resume Task materialization after canonical restore validation."""
    return _base_resume_task(
        checkpoint,
        target_task_id=target_task_id,
        target_tenant_id=target_tenant_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
    )


def _base_resume_task(
    checkpoint: TaskCheckpoint,
    *,
    target_task_id: str,
    target_tenant_id: str,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
) -> Task:
    assert_checkpoint_resume_materialization_eligible(
        checkpoint,
        target_task_id=target_task_id,
        target_tenant_id=target_tenant_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
    )
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
