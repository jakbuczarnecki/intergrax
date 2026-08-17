# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, status

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from dispute_sim_application.serving.schemas import DisputeSimRunRequestV1, DisputeSimRunResponseV1


@dataclass
class DisputeSimRunService:
    task_runner: UnifiedTaskRunner
    default_agent_id: str

    @classmethod
    def from_nexus_loop(
        cls,
        nexus_loop: NexusLoop,
        *,
        default_agent_id: str,
    ) -> DisputeSimRunService:
        return cls(
            task_runner=UnifiedTaskRunner(nexus_loop),
            default_agent_id=default_agent_id,
        )

    async def run_task(self, body: DisputeSimRunRequestV1) -> DisputeSimRunResponseV1:
        task_id, run_id = mint_intake_execution_identity()
        task = Task(
            task_id=task_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(capability=body.capability or "dispute.intake"),
            metadata=dict(body.metadata),
        )
        result = await self.task_runner.run_task(task, run_id=run_id)
        return DisputeSimRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_dispute_sim_routes(
    app: FastAPI,
    *,
    nexus_loop: NexusLoop,
    prefix: str = "/v1/dispute_sim",
    default_agent_id: str = "echo",
) -> DisputeSimRunService:
    service = DisputeSimRunService.from_nexus_loop(
        nexus_loop,
        default_agent_id=default_agent_id,
    )
    router = APIRouter(prefix=prefix, tags=["dispute_sim"])

    @router.post("/run", response_model=DisputeSimRunResponseV1)
    async def run_agent(body: DisputeSimRunRequestV1) -> DisputeSimRunResponseV1:
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
