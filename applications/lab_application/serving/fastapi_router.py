# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from intergrax.applications._shared.harness_auth import require_harness_api_key
from pydantic import BaseModel, Field

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.applications._shared.task_intake import (
    apply_long_running_enabled,
    apply_orchestration_graph_id,
)
from intergrax.runtime.task.task_run_bridge import new_run_id


class LabRunRequestV1(BaseModel):
    tenant_id: str = "lab"
    user_id: str = "lab-user"
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    graph_id: Optional[str] = None
    long_running: bool = False


class LabRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LabRunService:
    host_execution: HostTaskExecutionPort
    task_enricher: Callable[[Task], Task] | None = None

    @classmethod
    def from_host_execution(
        cls,
        host_execution: HostTaskExecutionPort,
        *,
        task_enricher: Callable[[Task], Task] | None = None,
    ) -> LabRunService:
        return cls(
            host_execution=host_execution,
            task_enricher=task_enricher,
        )

    async def run_task(self, body: LabRunRequestV1) -> LabRunResponseV1:
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
        task = apply_long_running_enabled(
            task,
            enabled=body.long_running,
            checkpoint_on_pause=True,
        )
        task = apply_orchestration_graph_id(task, body.graph_id)
        if self.task_enricher is not None:
            task = self.task_enricher(task)
        result = await self.host_execution.execute(task)
        return LabRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_lab_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    prefix: str = "/v1/lab",
    task_enricher: Callable[[Task], Task] | None = None,
) -> LabRunService:
    nexus_loop = host_execution.nexus_loop
    service = LabRunService.from_host_execution(
        host_execution,
        task_enricher=task_enricher,
    )
    router = APIRouter(
        prefix=prefix,
        tags=["lab"],
        dependencies=[Depends(require_harness_api_key)],
    )

    @router.post("/run", response_model=LabRunResponseV1)
    async def lab_run(body: LabRunRequestV1) -> LabRunResponseV1:
        try:
            return await service.run_task(body)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"lab_run_error: {exc.__class__.__name__}",
            ) from exc

    @router.get("/integrations/docling/health")
    async def docling_server_health() -> dict[str, object]:
        from intergrax.integrations.providers.document_parser.docling.config import (
            DoclingIntegrationConfig,
            DoclingMode,
        )
        from intergrax.integrations.providers.document_parser.docling.opens import check_docling_server_health

        config = DoclingIntegrationConfig.from_env()
        if config.mode is not DoclingMode.SERVER:
            return {
                "ok": config.mode is DoclingMode.LOCAL,
                "mode": config.mode.value,
                "detail": "server health applies only when INTERGRAX_DOCLING_MODE=server",
            }
        return check_docling_server_health(config)

    @router.get("/agents")
    async def lab_agents() -> dict[str, list[dict[str, object]]]:
        agents: list[dict[str, object]] = []
        for agent_id in nexus_loop.registry.list_agent_ids():
            agent = nexus_loop.registry.get(agent_id)
            contract = agent.get_contract()
            agents.append(
                {
                    "agent_id": contract.id,
                    "name": contract.name,
                    "capabilities": list(contract.capabilities),
                }
            )
        return {"agents": agents}

    app.include_router(router)
    return service
