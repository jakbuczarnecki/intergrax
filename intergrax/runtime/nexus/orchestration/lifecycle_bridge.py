# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Lifecycle and trace finalize helpers for NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_cost import aggregate_execution_metrics
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.governance_hooks import hook_context_for_task, run_hook_pair
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import (
    PersistingTaskTraceEmitter,
    TaskTraceEmitter,
    lifecycle_with_persisting_trace,
    lifecycle_with_trace,
)


def resolve_nexus_lifecycle(
    task: Task,
    *,
    lifecycle: Optional[TaskLifecycle],
    trace_emitter: Optional[TaskTraceEmitter],
    trace_store: Optional[RunTraceWriter],
    event_bus: RuntimeEventBus,
) -> tuple[TaskLifecycle, TaskTraceEmitter]:
    if lifecycle is not None:
        emitter = trace_emitter or TaskTraceEmitter(
            run_id=task.task_id,
            event_bus=event_bus,
        )
        return lifecycle, emitter
    if trace_store is not None:
        return lifecycle_with_persisting_trace(
            run_id=task.task_id,
            trace_store=trace_store,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            session_id=task.session_id or "",
            event_bus=event_bus,
        )
    return lifecycle_with_trace(run_id=task.task_id, event_bus=event_bus)


async def finalize_persisting_trace(
    trace_emitter: PersistingTaskTraceEmitter,
    executions: List[AgentExecutionResult],
    *,
    task_id: str,
    middleware: MiddlewarePipeline,
) -> None:
    ctx = hook_context_for_task(
        task_id=task_id or trace_emitter._run_id,
        run_id=trace_emitter._run_id,
        phase=ExecutionPhase.TRACE_PERSISTENCE,
    )
    await run_hook_pair(
        middleware,
        HookPoint.BEFORE_TRACE_PERSIST,
        HookPoint.AFTER_TRACE_PERSIST,
        ctx,
    )
    metrics = aggregate_execution_metrics(executions)
    trace_emitter.finalize(
        duration_ms=metrics.duration_ms,
        llm_usage=metrics.as_llm_usage(),
    )
