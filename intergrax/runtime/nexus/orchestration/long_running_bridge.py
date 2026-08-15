# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Long-running checkpoint helpers for NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.task.task import Task


class RuntimeEventPublisher(Protocol):
    async def __call__(
        self, event: RuntimeEvent, *, task: Optional[Task] = None
    ) -> None: ...


async def maybe_restore_long_running(
    task: Task,
    *,
    checkpoint_store: Optional[SQLiteTaskCheckpointStore],
    publish: RuntimeEventPublisher,
    notification_adapter: Optional[NotificationAdapter],
    run_id: str,
) -> None:
    if checkpoint_store is None:
        return
    restored = LongRunningCoordinator.restore_if_resuming(task, checkpoint_store)
    if restored is None:
        return
    await publish(
        runtime_event_from_task_state(
            task,
            run_id=run_id,
            message="long-running task restored from checkpoint",
        ).model_copy(
            update={
                "event_type": RuntimeEventType.RESUMED,
                "phase": ExecutionPhase.HUMAN_APPROVAL,
                "payload": {
                    "checkpoint_id": restored.checkpoint_id,
                    "resume_token": restored.resume_token,
                },
            }
        ),
        task=task,
    )
    await LongRunningCoordinator.notify_progress(
        task,
        subject="Task resumed",
        body=restored.progress_message or "checkpoint restored",
        adapter=notification_adapter,
    )


async def maybe_checkpoint_long_running(
    task: Task,
    *,
    checkpoint_store: Optional[SQLiteTaskCheckpointStore],
    publish: RuntimeEventPublisher,
    notification_adapter: Optional[NotificationAdapter],
    progress_message: str,
    run_id: str,
    attempt_id: str,
    plan: Optional[NexusPlan] = None,
    graph: Optional[object] = None,
    last_execution: Optional[AgentExecutionResult] = None,
) -> None:
    if checkpoint_store is None or not LongRunningCoordinator.should_checkpoint(task):
        return
    graph_obj = graph if isinstance(graph, ExecutionGraph) else None
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        task,
        checkpoint_store,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_message=progress_message,
        plan=plan,
        graph=graph_obj,
        last_execution=last_execution,
    )
    from intergrax.runtime.long_running.partial_results import partial_result_from_checkpoint

    partial = partial_result_from_checkpoint(checkpoint)
    await publish(
        runtime_event_from_task_state(
            task,
            run_id=run_id,
            message="long-running checkpoint saved",
        ).model_copy(
            update={
                "event_type": RuntimeEventType.PAUSED,
                "phase": ExecutionPhase.HUMAN_APPROVAL,
                "payload": {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "resume_token": checkpoint.resume_token,
                    "progress_message": progress_message,
                },
            }
        ),
        task=task,
    )
    await publish(
        runtime_event_from_task_state(
            task,
            run_id=run_id,
            message=progress_message or "task progress",
        ).model_copy(
            update={
                "event_type": RuntimeEventType.TASK_PROGRESS,
                "phase": ExecutionPhase.STEP_EXECUTION,
                "payload": partial.model_dump(mode="json"),
            }
        ),
        task=task,
    )
    if task.runtime.governance.human_request is not None:
        await LongRunningCoordinator.notify_hitl_pause(
            task,
            progress_message=progress_message,
            adapter=notification_adapter,
        )
    else:
        await LongRunningCoordinator.notify_partial_result(
            task,
            progress_message=progress_message,
            partial_payload=partial.partial_payload,
            last_step_summary=partial.last_step_summary,
            adapter=notification_adapter,
        )
