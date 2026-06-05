# © Artur Czarnecki. All rights reserved.

"""Minimal lab FastAPI surface for :class:`~intergrax.harness.app.HarnessApplication`."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class HarnessRunRequestV1(BaseModel):
    tenant_id: str = "lab"
    user_id: str = "lab-user"
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


class HarnessRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    metadata: dict[str, object] = Field(default_factory=dict)


def mount_harness_routes(
    app: FastAPI,
    *,
    nexus_loop: NexusLoop,
    prefix: str,
) -> UnifiedTaskRunner:
    task_runner = UnifiedTaskRunner(nexus_loop)
    router = APIRouter(prefix=prefix, tags=["harness"])

    @router.post("/run", response_model=HarnessRunResponseV1)
    async def run_agent(body: HarnessRunRequestV1) -> HarnessRunResponseV1:
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(capability=body.capability),
            metadata=dict(body.metadata),
        )
        try:
            result = await task_runner.run_task(task)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"run_error: {exc.__class__.__name__}",
            ) from exc
        return HarnessRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )

    @router.get("/agents")
    async def list_agents() -> dict[str, list[dict[str, object]]]:
        agents: list[dict[str, object]] = []
        for agent_id in nexus_loop.registry.list_agent_ids():
            contract = nexus_loop.registry.get(agent_id).get_contract()
            agents.append(
                {
                    "agent_id": contract.id,
                    "name": contract.name,
                    "capabilities": list(contract.capabilities),
                }
            )
        return {"agents": agents}

    app.include_router(router)
    return task_runner


def create_lab_fastapi_from_runtime(
    runtime: HarnessHostRuntime,
    *,
    route_prefix: str,
    mount_routes: bool = True,
) -> FastAPI:
    app = FastAPI(title=runtime.manifest.name)
    platform = bootstrap_nexus_platform(
        runtime.nexus_loop,
        trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
    )
    if mount_routes:
        mount_harness_routes(app, nexus_loop=runtime.nexus_loop, prefix=route_prefix)
    attach_plugin_shutdown(app, platform.shutdown_callbacks)
    return app
