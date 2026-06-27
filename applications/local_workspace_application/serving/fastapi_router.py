# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, status

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata
from local_workspace_application.serving.schemas import LocalWorkspaceRunRequestV1, LocalWorkspaceRunResponseV1


@dataclass
class LocalWorkspaceRunService:
    task_runner: UnifiedTaskRunner
    default_agent_id: str

    @classmethod
    def from_nexus_loop(
        cls,
        nexus_loop: NexusLoop,
        *,
        default_agent_id: str,
    ) -> LocalWorkspaceRunService:
        return cls(
            task_runner=UnifiedTaskRunner(nexus_loop),
            default_agent_id=default_agent_id,
        )

    async def run_task(self, body: LocalWorkspaceRunRequestV1) -> LocalWorkspaceRunResponseV1:
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(capability=body.capability or "local.workspace.search"),
            metadata=dict(body.metadata),
        )
        result = await self.task_runner.run_task(task)
        metadata = dict(result.metadata)
        attach_lkw_evidence_metadata(
            metadata,
            task_result=result,
            capability=body.capability or "local.workspace.search",
        )
        return LocalWorkspaceRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=metadata,
        )


def mount_local_workspace_routes(
    app: FastAPI,
    *,
    nexus_loop: NexusLoop,
    prefix: str = "/v1/local_workspace",
    default_agent_id: str = "local_search",
) -> LocalWorkspaceRunService:
    service = LocalWorkspaceRunService.from_nexus_loop(
        nexus_loop,
        default_agent_id=default_agent_id,
    )
    router = APIRouter(prefix=prefix, tags=["local_workspace"])

    @router.post("/run", response_model=LocalWorkspaceRunResponseV1)
    async def run_agent(body: LocalWorkspaceRunRequestV1) -> LocalWorkspaceRunResponseV1:
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
