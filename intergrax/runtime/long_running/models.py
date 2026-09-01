# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Long-running task models (§26, §42.9)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.task.task_state import TaskState


class TaskCheckpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: f"ckpt_{uuid4().hex[:16]}")
    task_id: str
    tenant_id: str
    resume_token: str
    task_state: TaskState
    task_snapshot: Dict[str, Any] = Field(default_factory=dict)
    progress_message: str = ""
    notify_channel: Optional[str] = None
    created_at_utc: str = ""
    schema_version: str = "task_checkpoint.v1"
    runtime: Optional[RuntimeCheckpoint] = None


__all__ = ["NotificationMessage", "TaskCheckpoint"]
