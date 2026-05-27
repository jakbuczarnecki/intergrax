# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.agents.agent_engine import AgentEngine
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.task_classifier import TaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter, lifecycle_with_trace


class NexusLoop:
    """
    Minimal global Nexus loop (§9.1, §41).

    Task → classify → plan → select agent → execute → validate → return TaskResult.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        classifier: Optional[TaskClassifier] = None,
        lifecycle: Optional[TaskLifecycle] = None,
        trace_emitter: Optional[TaskTraceEmitter] = None,
    ) -> None:
        self._registry = registry
        self._engine = AgentEngine(registry)
        self._classifier = classifier or TaskClassifier()
        self._router = AgentRouter(registry)
        self._lifecycle = lifecycle
        self._trace_emitter = trace_emitter

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def trace_emitter(self) -> Optional[TaskTraceEmitter]:
        return self._trace_emitter

    async def handle_task(self, task: Task) -> TaskResult:
        lifecycle, trace_emitter = self._resolve_lifecycle(task)
        self._trace_emitter = trace_emitter

        task = self._classifier.classify(task)
        lifecycle.transition(task, TaskState.CLASSIFIED)
        lifecycle.transition(task, TaskState.PLANNED)

        agent = self._router.route(task)
        contract = agent.get_contract()
        task.agent_id = contract.id
        lifecycle.transition(task, TaskState.RUNNING)

        request = task.to_runtime_request()
        execution = await self._engine.run_agent_with_result(agent, request)

        lifecycle.transition(task, TaskState.VALIDATING)
        final_state = (
            TaskState.COMPLETED
            if execution.status.value == "completed"
            else TaskState.FAILED
        )
        lifecycle.transition(task, final_state)

        return TaskResult(
            task_id=task.task_id,
            run_id=execution.run_id,
            state=task.state,
            answer=execution.summary,
            agent_id=contract.id,
            execution_result=execution,
            metadata={
                "validation_valid": final_state == TaskState.COMPLETED,
                "validation_errors": execution.errors,
                "task_trace_events": len(trace_emitter.events),
            },
        )

    def _resolve_lifecycle(self, task: Task) -> tuple[TaskLifecycle, TaskTraceEmitter]:
        if self._lifecycle is not None:
            emitter = self._trace_emitter or TaskTraceEmitter(run_id=task.task_id)
            return self._lifecycle, emitter
        return lifecycle_with_trace(run_id=task.task_id)
