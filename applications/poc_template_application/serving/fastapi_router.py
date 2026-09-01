# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id


class PocTemplateRunRequestV1(BaseModel):
    tenant_id: str = "lab"
    user_id: str = "lab-user"
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PocTemplateRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class PocTemplateRunService:
    host_execution: HostTaskExecutionPort

    @classmethod
    def from_host_execution(
        cls,
        host_execution: HostTaskExecutionPort,
    ) -> PocTemplateRunService:
        return cls(host_execution=host_execution)

    async def run_task(self, body: PocTemplateRunRequestV1) -> PocTemplateRunResponseV1:
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
        result = await self.host_execution.execute(task)
        return PocTemplateRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_poc_template_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    prefix: str = "/v1/poc_template",
) -> PocTemplateRunService:
    nexus_loop = host_execution.nexus_loop
    service = PocTemplateRunService.from_host_execution(host_execution)
    router = APIRouter(prefix=prefix, tags=["poc_template"])

    @router.post("/run", response_model=PocTemplateRunResponseV1)
    async def run_agent(body: PocTemplateRunRequestV1) -> PocTemplateRunResponseV1:
        try:
            return await service.run_task(body)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"run_error: {exc.__class__.__name__}",
            ) from exc

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
    return service
