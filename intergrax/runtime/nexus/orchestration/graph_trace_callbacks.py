# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Graph execution trace callbacks (Phase Q-N.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import TaskTraceEmitter


@dataclass(frozen=True, slots=True)
class GraphTraceCallbacks:
    """Strongly typed callbacks for ``GraphExecutor`` trace emission."""

    task: Task
    trace_emitter: TaskTraceEmitter

    def on_retry(self, record: RetryRecord) -> None:
        self.trace_emitter.emit(
            self.task,
            message=(
                f"retry attempt {record.attempt}: {record.reason} "
                f"-> {record.alternate_agent_id}"
            ),
        )

    def on_node_start(self, node: ExecutionNode) -> None:
        self.trace_emitter.emit(
            self.task,
            message=f"graph node start: {node.node_id}",
        )

    def on_node_complete(self, node: ExecutionNode) -> None:
        self.trace_emitter.emit(
            self.task,
            message=f"graph node complete: {node.node_id} status={node.status.value}",
        )
