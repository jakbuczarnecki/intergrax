# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Long-running checkpoint helpers for NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Protocol

from intergrax.contracts.execution_identity import (
    peek_active_execution_identity,
    validate_attempt_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
    TaskCheckpointReader,
)
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
    checkpoint_store: TaskCheckpointReader | None,
    publish: RuntimeEventPublisher,
    notification_adapter: Optional[NotificationAdapter],
    run_id: str,
    execution_terminal: ExecutionTerminalService | None = None,
) -> None:
    if checkpoint_store is None:
        return
    restored = LongRunningCoordinator.restore_if_resuming(
        task,
        checkpoint_store,
        execution_terminal=execution_terminal,
    )
    if restored is None:
        return
    if restored.runtime is not None:
        resolved_attempt_id = restored.runtime.attempt_id
    else:
        active_identity = peek_active_execution_identity()
        if active_identity is None:
            raise RuntimeError("attempt_id required for long-running restore event")
        resolved_attempt_id = active_identity[1]
    await publish(
        runtime_event_from_task_state(
            task,
            run_id=run_id,
            attempt_id=resolved_attempt_id,
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
    checkpoint_store: TaskCheckpointPersistence | None,
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
            attempt_id=attempt_id,
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
            attempt_id=attempt_id,
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
