# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Debug API progress aggregation for long-running tasks (§26, J.5)."""

from __future__ import annotations

from typing import Optional

from intergrax.debug.models import PartialResultItem, TaskProgressResponse
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.long_running.partial_results import build_task_progress_view
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointReader


class TaskProgressService:
    """Read-only progress view over checkpoints and optional runtime events."""

    def __init__(
        self,
        checkpoint_reader: TaskCheckpointReader,
        *,
        runtime_event_store: Optional[RuntimeEventPersistence] = None,
    ) -> None:
        self._checkpoints = checkpoint_reader
        self._events = runtime_event_store

    def get_progress(self, task_id: str, tenant_id: str) -> TaskProgressResponse:
        rows = self._checkpoints.list_for_task(task_id, tenant_id)
        if not rows and hasattr(self._checkpoints, "get_latest"):
            latest = self._checkpoints.get_latest(task_id, tenant_id)
            if latest is not None:
                rows = [latest]

        view = build_task_progress_view(
            task_id=task_id,
            tenant_id=tenant_id,
            checkpoints=rows,
        )
        progress_event_count = 0
        if self._events is not None:
            events = self._events.list_for_task(task_id, tenant_id=tenant_id, limit=500)
            progress_event_count = sum(
                1 for event in events if event.event_type == RuntimeEventType.TASK_PROGRESS
            )

        partial_items = [
            PartialResultItem(
                checkpoint_id=item.checkpoint_id,
                progress_message=item.progress_message,
                task_state=item.task_state,
                created_at_utc=item.created_at_utc,
                uaep_step_index=item.uaep_step_index,
                uaep_step_id=item.uaep_step_id,
                last_step_summary=item.last_step_summary,
                partial_payload=dict(item.partial_payload),
            )
            for item in view["partial_results"]
        ]
        latest = view.get("latest_partial_result")
        latest_item = None
        if latest is not None:
            latest_item = PartialResultItem(
                checkpoint_id=latest.checkpoint_id,
                progress_message=latest.progress_message,
                task_state=latest.task_state,
                created_at_utc=latest.created_at_utc,
                uaep_step_index=latest.uaep_step_index,
                uaep_step_id=latest.uaep_step_id,
                last_step_summary=latest.last_step_summary,
                partial_payload=dict(latest.partial_payload),
            )

        return TaskProgressResponse(
            task_id=task_id,
            tenant_id=tenant_id,
            task_state=str(view["task_state"]),
            progress_message=str(view["progress_message"]),
            resume_token=view.get("resume_token"),
            checkpoint_id=view.get("checkpoint_id"),
            notify_channel=view.get("notify_channel"),
            human_request_expires_at=view.get("human_request_expires_at"),
            is_paused=bool(view["is_paused"]),
            checkpoint_count=int(view["checkpoint_count"]),
            progress_event_count=progress_event_count,
            partial_results=partial_items,
            latest_partial_result=latest_item,
        )
