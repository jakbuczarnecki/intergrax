# © Artur Czarnecki. All rights reserved.

"""Shared harness HTTP routes for task control (FLOW-CTL.2, REL-ADV.4)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from intergrax.applications._shared.async_task_dispatch import get_async_status, run_async
from intergrax.applications._shared.harness_auth import require_harness_api_key
from intergrax.applications._shared.task_control import (
    cancel_active_task,
    set_task_autonomy,
)
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class HarnessAsyncRunRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str = "user"
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessAutonomyRequest(BaseModel):
    autonomy_level: AutonomyLevel


class HarnessResumeRequest(BaseModel):
    tenant_id: str = "default"
    resume_token: str = Field(min_length=1)
    operator_input: dict[str, Any] = Field(default_factory=dict)


class HarnessTaskControlResponse(BaseModel):
    task_id: str
    action: str
    accepted: bool
    detail: str = ""
    state: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def mount_harness_task_routes(
    app: FastAPI,
    *,
    task_runner: UnifiedTaskRunner,
    prefix: str = "/v1/tasks",
    checkpoint_store: TaskCheckpointPersistence | None = None,
    task_enricher: Callable[[Task], Task] | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix=prefix,
        tags=["harness-tasks"],
        dependencies=[Depends(require_harness_api_key)],
    )

    @router.post("/run-async")
    async def run_async_route(body: HarnessAsyncRunRequest) -> dict[str, Any]:
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            message=body.message,
            context=TaskContext(capability=body.capability),
            metadata=dict(body.metadata),
        )
        if task_enricher is not None:
            task = task_enricher(task)
        return await run_async(task_runner, task)

    @router.get("/{task_id}/status")
    async def task_status(task_id: str) -> dict[str, Any]:
        return await get_async_status(task_id)

    @router.post("/{task_id}/cancel", response_model=HarnessTaskControlResponse)
    async def cancel_task(task_id: str) -> HarnessTaskControlResponse:
        result = await cancel_active_task(task_id)
        if not result.accepted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.detail)
        return HarnessTaskControlResponse(
            task_id=result.task_id,
            action=result.action,
            accepted=result.accepted,
            detail=result.detail,
            state=result.state,
            metadata=dict(result.metadata or {}),
        )

    @router.post("/{task_id}/autonomy", response_model=HarnessTaskControlResponse)
    async def set_autonomy(task_id: str, body: HarnessAutonomyRequest) -> HarnessTaskControlResponse:
        result = await set_task_autonomy(task_id, body.autonomy_level)
        if not result.accepted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.detail)
        return HarnessTaskControlResponse(
            task_id=result.task_id,
            action=result.action,
            accepted=result.accepted,
            detail=result.detail,
            state=result.state,
            metadata=dict(result.metadata or {}),
        )

    @router.post("/{task_id}/resume")
    async def resume_task(task_id: str, body: HarnessResumeRequest) -> dict[str, Any]:
        if checkpoint_store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="checkpoint_store_not_configured",
            )
        checkpoint = checkpoint_store.get_by_token(
            task_id,
            body.tenant_id,
            body.resume_token,
        )
        if checkpoint is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="invalid_resume_token")
        from intergrax.applications._shared.task_control import resume_task_with_token

        result = await resume_task_with_token(
            task_runner,
            task_id=task_id,
            resume_token=body.resume_token,
            operator_input=body.operator_input,
            checkpoint=checkpoint,
        )
        return {
            "task_id": result.task_id,
            "state": result.state.value,
            "answer": result.answer,
            "resume_token": result.summary.resume_token,
        }

    app.include_router(router)
    return router
