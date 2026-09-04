# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id


class IntergraxAssistantRunRequestV1(BaseModel):
    tenant_id: str = "lab"
    user_id: str = "lab-user"
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntergraxAssistantRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class IntergraxAssistantRunService:
    host_execution: HostTaskExecutionPort

    @classmethod
    def from_host_execution(
        cls,
        host_execution: HostTaskExecutionPort,
    ) -> IntergraxAssistantRunService:
        return cls(host_execution=host_execution)

    async def run_task(self, body: IntergraxAssistantRunRequestV1) -> IntergraxAssistantRunResponseV1:
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
        return IntergraxAssistantRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_intergrax_assistant_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    prefix: str = "/v1/intergrax_assistant",
) -> IntergraxAssistantRunService:
    service = IntergraxAssistantRunService.from_host_execution(host_execution)
    router = APIRouter(prefix=prefix, tags=["intergrax_assistant"])

    @router.post("/run", response_model=IntergraxAssistantRunResponseV1)
    async def run_agent(body: IntergraxAssistantRunRequestV1) -> IntergraxAssistantRunResponseV1:
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
        for agent_id in registry.list_agent_ids():
            contract = registry.get(agent_id).get_contract()
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
