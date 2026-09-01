# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, status

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity
from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1, GovernedContractorRunResponseV1


@dataclass
class GovernedContractorRunService:
    host_execution: HostTaskExecutionPort
    default_agent_id: str

    @classmethod
    def from_host_execution(
        cls,
        host_execution: HostTaskExecutionPort,
        *,
        default_agent_id: str,
    ) -> GovernedContractorRunService:
        return cls(
            host_execution=host_execution,
            default_agent_id=default_agent_id,
        )

    async def run_task(self, body: GovernedContractorRunRequestV1) -> GovernedContractorRunResponseV1:
        task_id, _ = mint_intake_execution_identity()
        task = Task(
            task_id=task_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            agent_id=self.default_agent_id,
            message=body.message,
            context=TaskContext(capability=body.capability or "external_contractor.adapt"),
            metadata=dict(body.metadata),
        )
        result = await self.host_execution.execute(task)
        return GovernedContractorRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_governed_contractor_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    prefix: str = "/v1/governed_contractor",
    default_agent_id: str = "external_contractor_adapter",
) -> GovernedContractorRunService:
    service = GovernedContractorRunService.from_host_execution(
        host_execution,
        default_agent_id=default_agent_id,
    )
    nexus_loop = host_execution.nexus_loop
    router = APIRouter(prefix=prefix, tags=["governed_contractor"])

    @router.post("/run", response_model=GovernedContractorRunResponseV1)
    async def run_agent(body: GovernedContractorRunRequestV1) -> GovernedContractorRunResponseV1:
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
