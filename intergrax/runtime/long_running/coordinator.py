# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Checkpoint save/restore for long-running Nexus tasks (Phase F.4)."""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_task,
    build_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.long_running.notification import NotificationAdapter, resolve_notification_adapter
from intergrax.runtime.notifications.templates.hitl import build_hitl_pause_notification_message
from intergrax.runtime.notifications.templates.partial_result import (
    build_partial_result_notification_message,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.task.task import Task, TaskState


class LongRunningCoordinator:
    """Persists checkpoints and restores tasks on resume."""

    @staticmethod
    def is_long_running(task: Task) -> bool:
        return task.options.long_running.enabled

    @staticmethod
    def should_checkpoint(task: Task) -> bool:
        return (
            LongRunningCoordinator.is_long_running(task)
            and task.options.long_running.checkpoint_on_pause
        )

    @staticmethod
    def restore_if_resuming(
        task: Task,
        store: SQLiteTaskCheckpointStore,
    ) -> Optional[TaskCheckpoint]:
        token = task.options.long_running.resume_token
        if not LongRunningCoordinator.is_long_running(task) or not token:
            return None

        checkpoint = store.get_by_token(task.task_id, task.tenant_id, token)
        if checkpoint is None:
            return None

        incoming_human = task.options.human.model_copy(deep=True)
        restored = Task.model_validate(checkpoint.task_snapshot)
        task.state = restored.state
        task.options = restored.options
        task.runtime = restored.runtime
        task.message = restored.message
        task.agent_id = restored.agent_id
        task.context = restored.context
        task.options.long_running.enabled = True
        task.options.long_running.resume_token = token
        if incoming_human.verdict is not None or incoming_human.response_text is not None:
            task.options.human = incoming_human
        task.runtime.orchestration.checkpoint_id = checkpoint.checkpoint_id
        task.runtime.orchestration.resume_token = checkpoint.resume_token
        task.runtime.orchestration.progress_message = checkpoint.progress_message
        if checkpoint.runtime is not None:
            apply_runtime_checkpoint_to_task(task, checkpoint.runtime)
        task.sync_metadata()
        return checkpoint

    @staticmethod
    def persist_checkpoint(
        task: Task,
        store: SQLiteTaskCheckpointStore,
        *,
        progress_message: str = "",
        plan: Optional[NexusPlan] = None,
        graph: Optional[ExecutionGraph] = None,
        last_execution: Optional[AgentExecutionResult] = None,
    ) -> TaskCheckpoint:
        runtime = build_runtime_checkpoint(
            task,
            plan=plan,
            graph=graph,
            last_execution=last_execution,
        )
        existing_token = task.runtime.orchestration.resume_token
        checkpoint = SQLiteTaskCheckpointStore.build_checkpoint(
            task,
            progress_message=progress_message,
            resume_token=existing_token,
            runtime=runtime,
        )
        store.save(checkpoint)
        task.runtime.orchestration.checkpoint_id = checkpoint.checkpoint_id
        task.runtime.orchestration.resume_token = checkpoint.resume_token
        task.runtime.orchestration.progress_message = progress_message or checkpoint.progress_message
        apply_runtime_checkpoint_to_task(task, runtime)
        task.sync_metadata()
        return checkpoint

    @staticmethod
    async def notify_progress(
        task: Task,
        *,
        subject: str,
        body: str,
        adapter: Optional[NotificationAdapter] = None,
        extra: Optional[dict] = None,
    ) -> None:
        if not LongRunningCoordinator.is_long_running(task):
            return
        channel = task.options.long_running.notify_channel or "log"
        notifier = adapter or resolve_notification_adapter(channel)
        await notifier.notify(
            NotificationMessage(
                channel=channel,
                subject=subject,
                body=body,
                task_id=task.task_id,
                tenant_id=task.tenant_id,
                metadata={
                    "task_state": task.state.value,
                    "resume_token": task.runtime.orchestration.resume_token,
                    "checkpoint_id": task.runtime.orchestration.checkpoint_id,
                    **(extra or {}),
                },
            )
        )

    @staticmethod
    async def notify_partial_result(
        task: Task,
        *,
        progress_message: str,
        partial_payload: Optional[dict] = None,
        last_step_summary: Optional[str] = None,
        adapter: Optional[NotificationAdapter] = None,
    ) -> None:
        if not LongRunningCoordinator.is_long_running(task):
            return
        channel = task.options.long_running.notify_channel or "log"
        notifier = adapter or resolve_notification_adapter(channel)
        message = build_partial_result_notification_message(
            task,
            progress_message=progress_message,
            channel=channel,
            partial_payload=partial_payload,
            last_step_summary=last_step_summary,
        )
        await notifier.notify(message)

    @staticmethod
    async def notify_hitl_pause(
        task: Task,
        *,
        progress_message: str,
        adapter: Optional[NotificationAdapter] = None,
    ) -> None:
        if not LongRunningCoordinator.is_long_running(task):
            return
        channel = task.options.long_running.notify_channel or "log"
        notifier = adapter or resolve_notification_adapter(channel)
        message = build_hitl_pause_notification_message(
            task,
            progress_message=progress_message,
            channel=channel,
        )
        await notifier.notify(message)

    @staticmethod
    def paused_states() -> frozenset[TaskState]:
        return frozenset(
            {
                TaskState.WAITING_FOR_HUMAN,
                TaskState.WAITING_FOR_RESOURCES,
                TaskState.NEEDS_MORE_INFORMATION,
            }
        )
