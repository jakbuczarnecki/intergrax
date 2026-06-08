# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Graph execution trace callbacks (Phase Q-N.1, OBS-BUS-3 typed payloads)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.nexus.tracing.graph_node_diag import (
    GRAPH_NODE_STEP_COMPLETE,
    GRAPH_NODE_STEP_START,
    GraphNodeDiagV1,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
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
        payload = GraphNodeDiagV1(
            node_id=node.node_id,
            status=node.status.value,
            agent_id=node.agent_id or self.task.agent_id or "",
            capability=node.capability or self.task.context.capability or "",
        )
        self.trace_emitter.emit_trace_step(
            self.task,
            component=TraceComponent.PLANNER,
            step=GRAPH_NODE_STEP_START,
            message=f"graph node start: {node.node_id}",
            level=TraceLevel.INFO,
            payload=payload,
            extra_tags={"node_id": node.node_id},
        )

    def on_node_complete(self, node: ExecutionNode) -> None:
        payload = GraphNodeDiagV1(
            node_id=node.node_id,
            status=node.status.value,
            agent_id=node.agent_id or self.task.agent_id or "",
            capability=node.capability or self.task.context.capability or "",
        )
        self.trace_emitter.emit_trace_step(
            self.task,
            component=TraceComponent.PLANNER,
            step=GRAPH_NODE_STEP_COMPLETE,
            message=f"graph node complete: {node.node_id} status={node.status.value}",
            level=TraceLevel.INFO,
            payload=payload,
            extra_tags={"node_id": node.node_id},
        )
