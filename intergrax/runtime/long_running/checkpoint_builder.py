# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build and apply RuntimeCheckpoint snapshots (Phase G.1–G.2, Appendix B.01–B.02)."""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_execution_id,
    peek_active_execution_id,
    validate_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeRecorder,
    ExecutionTreeSnapshot,
    build_execution_tree_resume_plan,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    PLAN_SNAPSHOT_KEY,
    PendingDecision,
    RuntimeCheckpoint,
    UaepStepOutput,
)
from intergrax.utils.time_provider import SystemTimeProvider
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution


def resolve_task_runtime_checkpoint(task: Task) -> RuntimeCheckpoint | None:
    return task.runtime.orchestration.runtime_checkpoint


def build_task_checkpoint(
    task: Task,
    *,
    progress_message: str = "",
    resume_token: Optional[str] = None,
    runtime: Optional[RuntimeCheckpoint] = None,
) -> TaskCheckpoint:
    token = resume_token or task.runtime.orchestration.resume_token or f"rt_{uuid4().hex[:20]}"
    return TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        resume_token=token,
        task_state=task.state,
        task_snapshot=task.model_dump(mode="json"),
        progress_message=progress_message,
        notify_channel=task.options.long_running.notify_channel,
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        runtime=runtime,
    )


def build_runtime_checkpoint(
    task: Task,
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    plan: Optional[NexusPlan] = None,
    graph: Optional[ExecutionGraph] = None,
    last_execution: Optional[AgentExecutionResult] = None,
    execution_tree: Optional[ExecutionTreeSnapshot] = None,
) -> RuntimeCheckpoint:
    task_id = validate_task_id(task.task_id)
    from_task = resolve_task_runtime_checkpoint(task)

    node_states: Dict[str, str] = {}
    prior_outputs: Dict[str, Dict[str, str]] = {}
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

    plan_snapshot = plan.model_dump(mode="json") if plan is not None else None
    graph_snapshot = graph.model_dump(mode="json") if graph is not None else None
    pending_decisions = _collect_pending_decisions(task, last_execution)
    resolved_tree = execution_tree
    if resolved_tree is None and from_task is not None:
        resolved_tree = from_task.execution_tree
    if resolved_tree is None:
        root_execution_id = peek_active_execution_id() or mint_execution_id()
        resolved_tree = ExecutionTreeRecorder.start_root(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root_execution_id,
        ).snapshot
    resolved_tree.validate_for_task(task_id=task_id, run_id=run_id)
    if resolved_tree.attempt_id != attempt_id:
        raise ValueError(
            "execution tree attempt_id mismatch with active checkpoint attempt: "
            f"{resolved_tree.attempt_id!r} != {attempt_id!r}"
        )

    base_fields: dict[str, object] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "execution_tree": resolved_tree,
        "plan_id": plan_id,
        "graph_id": graph_id,
        "graph_node_id": graph_node_id,
        "node_states": node_states,
        "prior_node_outputs": prior_outputs,
        "plan_snapshot": plan_snapshot,
        "graph_snapshot": graph_snapshot,
        "pending_decisions": pending_decisions,
        "agent_id": last_execution.agent_id if last_execution else task.agent_id,
        "uaep_step_index": 0,
        "paused_phase": ExecutionPhase.HUMAN_APPROVAL.value,
        "pending_human_request": (
            last_execution.human_request.model_dump()
            if last_execution and last_execution.human_request
            else None
        ),
    }
    if from_task is not None:
        merged = from_task.model_dump(mode="json")
        merged.update(base_fields)
        runtime = RuntimeCheckpoint.model_validate(merged)
    else:
        runtime = RuntimeCheckpoint.model_validate(base_fields)
    runtime.validate_canonical()
    return runtime


def apply_runtime_checkpoint_to_task(task: Task, runtime: RuntimeCheckpoint) -> None:
    runtime.validate_canonical()
    if runtime.plan_id:
        task.runtime.orchestration.plan_id = runtime.plan_id
    if runtime.graph_id:
        task.runtime.orchestration.graph_id = runtime.graph_id
    if runtime.plan_snapshot:
        task.metadata[PLAN_SNAPSHOT_KEY] = runtime.plan_snapshot
    task.runtime.orchestration.runtime_checkpoint = runtime
    task.sync_metadata()


def apply_runtime_checkpoint_to_graph(
    graph: ExecutionGraph,
    runtime: RuntimeCheckpoint,
    prior_outputs: Dict[str, AgentExecutionResult],
    *,
    run_id: RunId,
) -> None:
    runtime.validate_canonical()
    runtime.execution_tree.validate_for_task(
        task_id=validate_task_id(graph.task_id),
        run_id=run_id,
    )
    if runtime.graph_snapshot and not runtime.node_states:
        restored = ExecutionGraph.model_validate(runtime.graph_snapshot)
        for node in graph.nodes:
            source = restored.node_by_id(node.node_id)
            node.status = source.status
            if source.execution_result is not None:
                node.execution_result = source.execution_result

    for node in graph.nodes:
        status_raw = runtime.node_states.get(node.node_id)
        if status_raw:
            try:
                node.status = ExecutionNodeStatus(status_raw)
            except ValueError:
                pass
        tree_entry = runtime.execution_tree.entry_by_graph_node_id(node.node_id)
        if tree_entry is not None and tree_entry.prior_output is not None:
            restored = AgentExecutionResult(
                agent_id=tree_entry.prior_output.agent_id,
                run_id=run_id,
                status=_status_from_summary(tree_entry.prior_output.status),
                summary=tree_entry.prior_output.summary,
            )
            prior_outputs[node.node_id] = restored
            if tree_entry.status is ExecutionCheckpointStatus.COMPLETED:
                node.execution_result = restored
                node.status = ExecutionNodeStatus.COMPLETED
            continue
        prior = runtime.prior_node_outputs.get(node.node_id)
        if prior and node.node_id not in prior_outputs:
            restored = AgentExecutionResult(
                agent_id=str(prior.get("agent_id") or node.agent_id or ""),
                run_id=run_id,
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
    tree_entry = checkpoint.execution_tree.entry_by_graph_node_id(node.node_id)
    if tree_entry is not None:
        if tree_entry.status is not ExecutionCheckpointStatus.COMPLETED:
            return False
        if tree_entry.prior_output is None:
            return False
        return node.node_id in prior_outputs
    if node.status not in (ExecutionNodeStatus.COMPLETED, ExecutionNodeStatus.SKIPPED):
        return False
    return node.node_id in prior_outputs


def _status_from_summary(raw: str | None) -> AgentExecutionStatus:
    if raw == AgentExecutionStatus.NEEDS_INPUT.value:
        return AgentExecutionStatus.NEEDS_INPUT
    if raw == AgentExecutionStatus.FAILED.value:
        return AgentExecutionStatus.FAILED
    return AgentExecutionStatus.COMPLETED


def _collect_pending_decisions(
    task: Task,
    last_execution: Optional[AgentExecutionResult],
) -> List[PendingDecision]:
    pending: List[PendingDecision] = []
    if last_execution is not None and last_execution.human_request is not None:
        pending.append(
            PendingDecision(
                type="human_request",
                agent_id=last_execution.agent_id,
                payload=last_execution.human_request.model_dump(mode="json"),
            )
        )
    human_request = task.runtime.governance.human_request
    if human_request is not None:
        pending.append(
            PendingDecision(
                type="human_request",
                agent_id=task.agent_id or "",
                payload=human_request.model_dump(mode="json"),
            )
        )
    return pending


def should_skip_uaep_step(
    *,
    step_index: int,
    step_id: str,
    checkpoint: Optional[RuntimeCheckpoint],
    approval: HumanApprovalResolution | None,
) -> bool:
    if checkpoint is None or approval is None:
        return False
    if checkpoint.uaep_step_index != step_index:
        return False
    if checkpoint.uaep_step_id and checkpoint.uaep_step_id != step_id:
        return False
    if checkpoint.uaep_step_cursor is not None and not checkpoint.uaep_step_completed:
        return False
    return checkpoint.last_step_output is not None


def should_resume_uaep_step(
    *,
    step_index: int,
    step_id: str,
    checkpoint: Optional[RuntimeCheckpoint],
    approval: HumanApprovalResolution | None,
) -> bool:
    if checkpoint is None or approval is None:
        return False
    if checkpoint.uaep_step_index != step_index:
        return False
    if checkpoint.uaep_step_id and checkpoint.uaep_step_id != step_id:
        return False
    if checkpoint.uaep_step_completed:
        return False
    return checkpoint.uaep_step_cursor is not None


def prepare_task_for_checkpoint_resume(
    task: Task,
    checkpoint: TaskCheckpoint,
    *,
    active_attempt_id: AttemptId,
    active_root_execution_id: ExecutionId,
) -> RuntimeCheckpoint:
    runtime = checkpoint.runtime
    if runtime is None:
        raise ValueError(
            f"checkpoint {checkpoint.checkpoint_id!r} missing canonical execution identity"
        )
    run_id = runtime.run_id
    resume_plan = build_execution_tree_resume_plan(
        runtime.execution_tree,
        task_id=validate_task_id(task.task_id),
        run_id=run_id,
        new_attempt_id=active_attempt_id,
        new_root_execution_id=active_root_execution_id,
    )
    resumed_runtime = runtime.model_copy(
        update={
            "attempt_id": active_attempt_id,
            "execution_tree": resume_plan.active_snapshot,
        }
    )
    resumed_runtime.validate_canonical()
    apply_runtime_checkpoint_to_task(task, resumed_runtime)
    return resumed_runtime


def snapshot_active_execution_tree(
    recorder: ExecutionTreeRecorder,
) -> ExecutionTreeSnapshot:
    return recorder.snapshot


def sync_execution_tree_to_task(
    task: Task,
    recorder: ExecutionTreeRecorder,
) -> None:
    existing = resolve_task_runtime_checkpoint(task)
    if existing is None:
        return
    task.runtime.orchestration.runtime_checkpoint = existing.model_copy(
        update={"execution_tree": recorder.snapshot}
    )


def record_graph_node_completion(
    recorder: ExecutionTreeRecorder,
    *,
    execution_id: ExecutionId,
    node: ExecutionNode,
    execution: AgentExecutionResult,
) -> None:
    status = (
        ExecutionCheckpointStatus.FAILED
        if execution.status is AgentExecutionStatus.FAILED
        else ExecutionCheckpointStatus.COMPLETED
    )
    prior_output = ExecutionPriorOutput(
        agent_id=execution.agent_id,
        summary=execution.summary,
        status=execution.status.value,
        graph_node_id=node.node_id,
    )
    if status is ExecutionCheckpointStatus.FAILED:
        recorder.record_failed(execution_id, prior_output=prior_output)
    else:
        recorder.record_completed(execution_id, prior_output=prior_output)
