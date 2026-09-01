# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Partial result snapshots for long-running tasks (§26, J.5)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task_state import TaskState

_PAUSED_STATES = frozenset(
    {
        TaskState.WAITING_FOR_HUMAN.value,
        TaskState.WAITING_FOR_RESOURCES.value,
        TaskState.NEEDS_MORE_INFORMATION.value,
    }
)


class PartialResultSnapshot(BaseModel):
    """Serializable partial progress captured at a checkpoint."""

    checkpoint_id: str
    progress_message: str
    task_state: str
    created_at_utc: str
    uaep_step_index: Optional[int] = None
    uaep_step_id: Optional[str] = None
    last_step_summary: Optional[str] = None
    partial_payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "partial_result.v1"


def partial_result_from_checkpoint(checkpoint: TaskCheckpoint) -> PartialResultSnapshot:
    runtime = checkpoint.runtime
    partial_payload: Dict[str, Any] = {}
    last_step_summary: Optional[str] = None
    uaep_step_index: Optional[int] = None
    uaep_step_id: Optional[str] = None

    if runtime is not None:
        uaep_step_index = runtime.uaep_step_index
        uaep_step_id = runtime.uaep_step_id
        if runtime.last_step_output:
            last_step_summary = str(runtime.last_step_output.get("summary") or "") or None
            partial_payload["last_step_output"] = dict(runtime.last_step_output)
        if runtime.prior_node_outputs:
            partial_payload["prior_node_outputs"] = dict(runtime.prior_node_outputs)
        if runtime.graph_node_id:
            partial_payload["graph_node_id"] = runtime.graph_node_id
        if runtime.plan_id:
            partial_payload["plan_id"] = runtime.plan_id
        if runtime.graph_id:
            partial_payload["graph_id"] = runtime.graph_id

    return PartialResultSnapshot(
        checkpoint_id=checkpoint.checkpoint_id,
        progress_message=checkpoint.progress_message,
        task_state=checkpoint.task_state.value,
        created_at_utc=checkpoint.created_at_utc,
        uaep_step_index=uaep_step_index,
        uaep_step_id=uaep_step_id,
        last_step_summary=last_step_summary,
        partial_payload=partial_payload,
    )


def build_task_progress_view(
    *,
    task_id: str,
    tenant_id: str,
    checkpoints: List[TaskCheckpoint],
) -> Dict[str, Any]:
    """Aggregate checkpoint history into a debug/lab progress view."""
    partial_results = [partial_result_from_checkpoint(row) for row in checkpoints]
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    latest_partial = partial_results[-1] if partial_results else None

    task_state = TaskState.CREATED.value
    progress_message = ""
    resume_token: Optional[str] = None
    checkpoint_id: Optional[str] = None
    notify_channel: Optional[str] = None
    human_request_expires_at: Optional[str] = None

    if latest_checkpoint is not None:
        task_state = latest_checkpoint.task_state.value
        progress_message = latest_checkpoint.progress_message
        resume_token = latest_checkpoint.resume_token
        checkpoint_id = latest_checkpoint.checkpoint_id
        notify_channel = latest_checkpoint.notify_channel
        try:
            from intergrax.runtime.task.task import Task

            task = Task.model_validate(latest_checkpoint.task_snapshot)
            human_request_expires_at = task.runtime.governance.human_request_expires_at
            if not progress_message:
                progress_message = task.runtime.orchestration.progress_message
        except Exception:
            pass

    return {
        "task_id": task_id,
        "tenant_id": tenant_id,
        "task_state": task_state,
        "progress_message": progress_message,
        "resume_token": resume_token,
        "checkpoint_id": checkpoint_id,
        "notify_channel": notify_channel,
        "human_request_expires_at": human_request_expires_at,
        "is_paused": task_state in _PAUSED_STATES,
        "partial_results": partial_results,
        "latest_partial_result": latest_partial,
        "checkpoint_count": len(checkpoints),
    }
