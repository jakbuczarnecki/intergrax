# © Artur Czarnecki. All rights reserved.

"""Canonical lab harness entry for NexusLoop task execution (OBS-DIAG / qualification)."""

from __future__ import annotations

from typing import Optional, Protocol

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.execution_identity import AttemptId, RunId, mint_run_id
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext, TaskResult
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from testing_support.nexus_hitl_test_agent import prepare_nexus_hitl_resume_task


class _CheckpointReader(Protocol):
    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None: ...


def lab_request_identity_for_task(task: Task) -> RequestIdentity:
    principal = (task.user_id or "operator").strip()
    return RequestIdentity(
        tenant_id=task.tenant_id.strip(),
        user_id=principal or None,
        principal_type=PrincipalType.USER,
        auth_subject=principal or "operator",
    )


def prepare_lab_graph_task(task: Task) -> Task:
    """Ensure graph execution hydrates task memory via canonical request identity."""
    if task.canonical_identity is not None:
        return task
    return task.model_copy(
        update={"canonical_identity": lab_request_identity_for_task(task)}
    )


def build_lab_unified_task_runner(nexus_loop: NexusLoop) -> UnifiedTaskRunner:
    return UnifiedTaskRunner(
        nexus_loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )


async def run_lab_nexus_task(
    nexus_loop: NexusLoop,
    task: Task,
    *,
    run_id: Optional[RunId] = None,
    attempt_id: Optional[AttemptId] = None,
    resume_checkpoint: Optional[TaskCheckpoint] = None,
) -> TaskResult:
    """Execute a task through canonical root execution (identity + governance)."""
    runner = build_lab_unified_task_runner(nexus_loop)
    return await runner.run_task(
        task,
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id,
        resume_checkpoint=resume_checkpoint,
    )


def _default_response_text(verdict: HumanResponseVerdict) -> str:
    if verdict is HumanResponseVerdict.APPROVE:
        return "approve"
    if verdict is HumanResponseVerdict.REJECT:
        return "reject"
    if verdict is HumanResponseVerdict.ESCALATE:
        return "escalate"
    return verdict.value


async def resume_lab_nexus_hitl(
    nexus_loop: NexusLoop,
    *,
    paused: TaskResult,
    checkpoint_store: _CheckpointReader,
    tenant_id: str,
    user_id: str,
    message: str,
    capability: str,
    run_id: RunId,
    verdict: HumanResponseVerdict = HumanResponseVerdict.APPROVE,
    response_text: str | None = None,
) -> TaskResult:
    """Canonical durable HITL resume (approver evidence + checkpoint continuity)."""
    checkpoint = checkpoint_store.get_latest(paused.task_id, tenant_id)
    if checkpoint is None or checkpoint.runtime is None:
        raise AssertionError("HITL resume requires a persisted runtime checkpoint")
    resume_token = paused.summary.resume_token
    if not resume_token:
        raise AssertionError("HITL resume requires paused.summary.resume_token")
    resolved_text = response_text or _default_response_text(verdict)
    resume_task = Task(
        tenant_id=tenant_id,
        user_id=user_id,
        message=message,
        context=TaskContext(capability=capability),
        task_id=paused.task_id,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True, resume_token=resume_token
            ),
            human=TaskHumanInput(response_text=resolved_text, verdict=verdict),
        ),
        metadata={
            "human_response": resolved_text,
            "resume_token": resume_token,
            **(
                {"human_approved": True}
                if verdict is HumanResponseVerdict.APPROVE
                else {}
            ),
        },
    )
    if verdict is HumanResponseVerdict.APPROVE:
        prepare_nexus_hitl_resume_task(
            resume_task,
            loaded=checkpoint,
            run_id=run_id,
            human_approved=True,
        )
    elif verdict is HumanResponseVerdict.REJECT:
        prepare_nexus_hitl_resume_task(
            resume_task,
            loaded=checkpoint,
            run_id=run_id,
            human_rejected=True,
        )
    else:
        prepare_nexus_hitl_resume_task(
            resume_task,
            loaded=checkpoint,
            run_id=run_id,
        )
    return await run_lab_nexus_task(
        nexus_loop,
        resume_task,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )
