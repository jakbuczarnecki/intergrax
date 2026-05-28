# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build and apply RuntimeCheckpoint snapshots (Phase G.1–G.2)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    GraphNodeCheckpoint,
    RuntimeCheckpoint,
    attach_runtime_checkpoint_to_metadata,
    runtime_checkpoint_from_execution_structured,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNodeStatus
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.task.task import Task


def build_runtime_checkpoint(
    task: Task,
    *,
    plan: Optional[NexusPlan] = None,
    graph: Optional[ExecutionGraph] = None,
    last_execution: Optional[AgentExecutionResult] = None,
) -> RuntimeCheckpoint:
    from_execution = None
    if last_execution is not None:
        from_execution = runtime_checkpoint_from_execution_structured(last_execution.structured_data)

    node_states: Dict[str, str] = {}
    prior_outputs: Dict[str, Dict[str, Any]] = {}
    if graph is not None:
        for node in graph.nodes:
            node_states[node.node_id] = node.status.value
            if node.execution_result is not None:
                prior_outputs[node.node_id] = {
                    "agent_id": node.execution_result.agent_id,
                    "summary": node.execution_result.summary,
                    "status": node.execution_result.status.value,
                }

    plan_id = (plan.plan_id if plan else None) or task.runtime.orchestration.plan_id
    graph_id = (graph.graph_id if graph else None) or task.runtime.orchestration.graph_id
    graph_node_id = None
    if last_execution is not None and graph is not None:
        for node in graph.nodes:
            if node.execution_result is last_execution or (
                node.agent_id == last_execution.agent_id
                and node.status == ExecutionNodeStatus.PENDING
            ):
                graph_node_id = node.node_id
                break
        if graph_node_id is None and graph.nodes:
            graph_node_id = graph.nodes[-1].node_id

    if from_execution is not None:
        merged = from_execution.model_copy(
            update={
                "plan_id": from_execution.plan_id or plan_id,
                "graph_id": from_execution.graph_id or graph_id,
                "graph_node_id": from_execution.graph_node_id or graph_node_id,
                "node_states": from_execution.node_states or node_states,
                "prior_node_outputs": from_execution.prior_node_outputs or prior_outputs,
            }
        )
        return merged

    return RuntimeCheckpoint(
        plan_id=plan_id,
        graph_id=graph_id,
        graph_node_id=graph_node_id,
        agent_id=last_execution.agent_id if last_execution else task.agent_id,
        uaep_step_index=0,
        paused_phase=ExecutionPhase.HUMAN_APPROVAL.value,
        node_states=node_states,
        prior_node_outputs=prior_outputs,
        pending_human_request=(
            last_execution.human_request.model_dump()
            if last_execution and last_execution.human_request
            else None
        ),
    )


def apply_runtime_checkpoint_to_task(task: Task, runtime: RuntimeCheckpoint) -> None:
    if runtime.plan_id:
        task.runtime.orchestration.plan_id = runtime.plan_id
    if runtime.graph_id:
        task.runtime.orchestration.graph_id = runtime.graph_id
    attach_runtime_checkpoint_to_metadata(task.metadata, runtime)
    task.sync_metadata()


def apply_runtime_checkpoint_to_graph(
    graph: ExecutionGraph,
    runtime: RuntimeCheckpoint,
    prior_outputs: Dict[str, AgentExecutionResult],
) -> None:
    for node in graph.nodes:
        status_raw = runtime.node_states.get(node.node_id)
        if status_raw:
            try:
                node.status = ExecutionNodeStatus(status_raw)
            except ValueError:
                pass
        prior = runtime.prior_node_outputs.get(node.node_id)
        if prior and node.node_id not in prior_outputs:
            restored = AgentExecutionResult(
                agent_id=str(prior.get("agent_id") or node.agent_id or ""),
                run_id=task_run_id_placeholder(graph.task_id),
                status=_status_from_summary(prior.get("status")),
                summary=str(prior.get("summary") or ""),
            )
            prior_outputs[node.node_id] = restored
            if node.status in (
                ExecutionNodeStatus.COMPLETED,
                ExecutionNodeStatus.SKIPPED,
            ):
                node.execution_result = restored


def should_skip_graph_node(
    node: ExecutionNode,
    *,
    checkpoint: Optional[RuntimeCheckpoint],
    prior_outputs: Dict[str, AgentExecutionResult],
) -> bool:
    if checkpoint is None:
        return False
    if node.status not in (ExecutionNodeStatus.COMPLETED, ExecutionNodeStatus.SKIPPED):
        return False
    return node.node_id in prior_outputs


def task_run_id_placeholder(task_id: str) -> str:
    return task_id


def _status_from_summary(raw: Any) -> AgentExecutionStatus:
    if raw == AgentExecutionStatus.NEEDS_INPUT.value:
        return AgentExecutionStatus.NEEDS_INPUT
    if raw == AgentExecutionStatus.FAILED.value:
        return AgentExecutionStatus.FAILED
    return AgentExecutionStatus.COMPLETED


def should_skip_uaep_step(
    *,
    step_index: int,
    step_id: str,
    checkpoint: Optional[RuntimeCheckpoint],
    human_approved: bool,
) -> bool:
    if checkpoint is None or not human_approved:
        return False
    if checkpoint.uaep_step_index != step_index:
        return False
    if checkpoint.uaep_step_id and checkpoint.uaep_step_id != step_id:
        return False
    return checkpoint.last_step_output is not None
