# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Internal Execution Engine finalization callback typing (not a public platform seam)."""

from __future__ import annotations

from typing import List, Optional, Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.runtime_event_metric import RuntimeEventMetricScope
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_trace import TaskTraceEmitter


class NexusFinishTaskFn(Protocol):
    async def __call__(
        self,
        task: Task,
        trace_emitter: TaskTraceEmitter,
        *,
        answer: str,
        executions: List[AgentExecutionResult],
        validation: ValidationResult,
        plan: Optional[NexusPlan],
        retry_records: List[RetryRecord],
        graph_id: str,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult: ...
