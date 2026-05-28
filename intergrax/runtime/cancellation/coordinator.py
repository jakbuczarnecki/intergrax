# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cooperative task cancellation helpers (§42.26, Phase G.8)."""

from __future__ import annotations

from typing import Any, Dict, List

from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNodeStatus
from intergrax.runtime.task.task import Task

CANCELLATION_REQUESTED_KEY = "cancellation_requested"
CANCELLATION_REASON_KEY = "cancellation_reason"

__all__ = [
    "CANCELLATION_REASON_KEY",
    "CANCELLATION_REQUESTED_KEY",
    "CancellationCoordinator",
]


class CancellationCoordinator:
    """Request, detect and clear cooperative cancellation signals on tasks."""

    @staticmethod
    def is_requested(metadata: Dict[str, Any]) -> bool:
        return bool(metadata.get(CANCELLATION_REQUESTED_KEY))

    @staticmethod
    def request(task: Task, *, reason: str = "") -> None:
        task.metadata[CANCELLATION_REQUESTED_KEY] = True
        if reason:
            task.metadata[CANCELLATION_REASON_KEY] = reason
        task.sync_metadata()

    @staticmethod
    def clear(task: Task) -> None:
        task.metadata.pop(CANCELLATION_REQUESTED_KEY, None)
        task.metadata.pop(CANCELLATION_REASON_KEY, None)
        task.sync_metadata()

    @staticmethod
    def propagate(source: Dict[str, Any], target: Dict[str, Any]) -> None:
        if not CancellationCoordinator.is_requested(source):
            return
        target[CANCELLATION_REQUESTED_KEY] = True
        reason = source.get(CANCELLATION_REASON_KEY)
        if reason:
            target[CANCELLATION_REASON_KEY] = reason

    @staticmethod
    def mark_pending_graph_nodes_cancelled(graph: ExecutionGraph) -> List[str]:
        cancelled_ids: List[str] = []
        for node in graph.nodes:
            if node.status in (ExecutionNodeStatus.PENDING, ExecutionNodeStatus.RUNNING):
                node.status = ExecutionNodeStatus.SKIPPED
                node.metadata["cancelled"] = True
                cancelled_ids.append(node.node_id)
        return cancelled_ids

    @staticmethod
    def clear_checkpoint_state(task: Task) -> None:
        """Stub cleanup after cancel — drop resume pointers without deleting store rows."""
        task.runtime.orchestration.checkpoint_id = None
        task.runtime.orchestration.resume_token = None
        task.runtime.orchestration.progress_message = ""
        task.options.long_running.resume_token = None
        task.sync_metadata()
