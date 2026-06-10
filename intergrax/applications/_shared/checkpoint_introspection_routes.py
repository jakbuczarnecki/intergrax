# © Artur Czarnecki. All rights reserved.

"""Checkpoint introspection HTTP API for product hosts (AUDIT-IDEAL-8.2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from intergrax.applications._shared.harness_auth import require_harness_api_key
from intergrax.debug.progress_service import TaskProgressService
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointReader


def create_checkpoint_introspection_router(
    checkpoint_reader: TaskCheckpointReader,
    *,
    enabled: bool = True,
    prefix: str = "/v1/tasks",
) -> APIRouter:
    """Read-only checkpoint/progress introspection beyond lab debug surfaces."""
    router = APIRouter(
        prefix=prefix,
        tags=["harness-checkpoints"],
        dependencies=[Depends(require_harness_api_key)],
    )
    service = TaskProgressService(checkpoint_reader)

    @router.get("/{task_id}/checkpoints")
    def list_checkpoints(task_id: str, tenant_id: str = "default") -> dict[str, object]:
        if not enabled:
            raise HTTPException(status_code=404, detail="checkpoint introspection disabled")
        progress = service.get_progress(task_id, tenant_id)
        return progress.model_dump()

    return router
