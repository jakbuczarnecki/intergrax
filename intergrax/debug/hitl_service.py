# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab workflow: resume paused tasks via debug API (Phase G.6)."""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.long_running.resume_planner import execution_identity_from_checkpoint
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.contracts.human_approver import HumanApproverEvidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)


class DebugHitlResumeService:
    """Resumes long-running tasks using persisted checkpoints + human response."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        checkpoint_store: TaskCheckpointPersistence,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
        human_decision_store: HumanDecisionPersistence | None = None,
    ) -> None:
        self._registry = registry
        self._checkpoint_store = checkpoint_store
        self._runtime_event_store = runtime_event_store
        self._human_decision_store = human_decision_store

    async def resume_with_human_response(
        self,
        task_id: str,
        tenant_id: str,
        *,
        verdict: HumanResponseVerdict,
        response_text: str = "",
        resume_token: Optional[str] = None,
        approver: HumanApproverEvidence | None = None,
    ) -> TaskResult:
        checkpoint = self._resolve_checkpoint(task_id, tenant_id, resume_token)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for task {task_id!r} (tenant={tenant_id})")

        paused = Task.model_validate(checkpoint.task_snapshot)
        pause_record = paused.runtime.governance.pause_record
        if pause_record is None:
            raise ValueError(
                f"Checkpoint for task {task_id!r} has no active pause record for HITL resume"
            )

        task = Task.model_validate(checkpoint.task_snapshot)
        effective_approver = approver
        if effective_approver is None:
            from intergrax.contracts.human_approver import local_development_approver_evidence

            effective_approver = local_development_approver_evidence(tenant_id=tenant_id)
        task.options = TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
                notify_channel=checkpoint.notify_channel,
                checkpoint_on_pause=task.options.long_running.checkpoint_on_pause,
            ),
            human=TaskHumanInput(
                response_text=response_text or verdict.value,
                verdict=verdict.value,
                pause_id=pause_record.pause_id,
                human_request_id=pause_record.human_request_id,
                approver=effective_approver,
            ),
        )
        task.metadata["resume_token"] = checkpoint.resume_token
        task.sync_metadata()

        run_id, attempt_id = execution_identity_from_checkpoint(checkpoint)

        loop = NexusLoop(
            self._registry,
            checkpoint_store=self._checkpoint_store,
            runtime_event_store=self._runtime_event_store,
            human_decision_store=self._human_decision_store,
        )
        return await loop.handle_task(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
        )

    def _resolve_checkpoint(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: Optional[str],
    ):
        if resume_token:
            return self._checkpoint_store.get_by_token(task_id, tenant_id, resume_token)
        return self._checkpoint_store.get_latest(task_id, tenant_id)
