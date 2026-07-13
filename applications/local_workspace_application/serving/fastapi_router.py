# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, status

from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.proof_summary import attach_lkw_proof_summary_metadata
from local_workspace_application.serving.run_artifact_metadata import ensure_run_artifact_bundle_metadata
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata
from local_workspace_application.serving.runtime_event_metadata import attach_runtime_event_summary_metadata
from local_workspace_application.serving.schemas import LocalWorkspaceRunRequestV1, LocalWorkspaceRunResponseV1


@dataclass
class LocalWorkspaceRunService:
    task_executor: LocalWorkspaceTaskExecutor
    default_agent_id: str

    async def run_task(self, body: LocalWorkspaceRunRequestV1) -> LocalWorkspaceRunResponseV1:
        run_id = new_run_id()
        metadata = dict(body.metadata)
        if body.tenant_id and "tenant_id" not in metadata:
            metadata["tenant_id"] = body.tenant_id
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(capability=body.capability or "local.workspace.search"),
            metadata=metadata,
        )
        try:
            result = await self.task_executor.execute(task)
        except HostNotAcceptingWorkError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error_id": exc.error_id, "message": exc.detail},
            ) from exc
        metadata = dict(result.metadata)
        ensure_run_artifact_bundle_metadata(metadata, task_result=result)
        attach_lkw_evidence_metadata(
            metadata,
            task_result=result,
            capability=body.capability or "local.workspace.search",
        )
        attach_runtime_event_summary_metadata(
            metadata,
            task_result=result,
            nexus_loop=self.task_executor.nexus_loop,
            tenant_id=body.tenant_id or "default",
        )
        attach_lkw_proof_summary_metadata(
            metadata,
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
    task_executor: LocalWorkspaceTaskExecutor,
    prefix: str = "/v1/local_workspace",
    default_agent_id: str = "local_search",
) -> LocalWorkspaceRunService:
    service = LocalWorkspaceRunService(
        task_executor=task_executor,
        default_agent_id=default_agent_id,
    )
    router = APIRouter(prefix=prefix, tags=["local_workspace"])

    @router.post("/run", response_model=LocalWorkspaceRunResponseV1)
    async def run_agent(body: LocalWorkspaceRunRequestV1) -> LocalWorkspaceRunResponseV1:
        try:
            return await service.run_task(body)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"run_error: {exc.__class__.__name__}",
            ) from exc

    @router.get("/agents")
    async def list_agents() -> dict[str, list[dict[str, object]]]:
        agents: list[dict[str, object]] = []
        nexus_loop = task_executor.nexus_loop
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
