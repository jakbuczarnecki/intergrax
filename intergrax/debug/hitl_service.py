# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab workflow: resume paused tasks via debug API (Phase G.6)."""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions


class DebugHitlResumeService:
    """Resumes long-running tasks using persisted checkpoints + human response."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        checkpoint_store: TaskCheckpointPersistence,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
        human_decision_store: Optional[SQLiteHumanDecisionStore] = None,
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
        response: str,
        resume_token: Optional[str] = None,
        user_id: str = "debug_operator",
    ) -> TaskResult:
        checkpoint = self._resolve_checkpoint(task_id, tenant_id, resume_token)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for task {task_id!r} (tenant={tenant_id})")

        paused = Task.model_validate(checkpoint.task_snapshot)
        metadata: dict[str, str | bool] = {
            "human_response": response,
            "resume_token": checkpoint.resume_token,
        }
        normalized = response.strip().lower()
        if normalized in {"approve", "yes", "ok"}:
            metadata["human_approved"] = True

        task = Task(
            tenant_id=tenant_id,
            user_id=user_id or paused.user_id,
            message=paused.message,
            context=paused.context,
            task_id=checkpoint.task_id,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(
                    enabled=True,
                    resume_token=checkpoint.resume_token,
                    notify_channel=checkpoint.notify_channel,
                ),
            ),
            metadata=metadata,
        )
        HumanPauseCoordinator.record_human_response(task, response)

        loop = NexusLoop(
            self._registry,
            checkpoint_store=self._checkpoint_store,
            runtime_event_store=self._runtime_event_store,
            human_decision_store=self._human_decision_store,
        )
        return await loop.handle_task(task)

    def _resolve_checkpoint(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: Optional[str],
    ):
        if resume_token:
            return self._checkpoint_store.get_by_token(task_id, tenant_id, resume_token)
        return self._checkpoint_store.get_latest(task_id, tenant_id)
