# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from research_application.serving.schemas import ResearchRunRequestV1, ResearchRunResponseV1


@dataclass
class ResearchRunService:
    task_runner: UnifiedTaskRunner

    @classmethod
    def from_nexus_loop(cls, nexus_loop: NexusLoop) -> ResearchRunService:
        return cls(task_runner=UnifiedTaskRunner(nexus_loop))

    async def run_pipeline(self, body: ResearchRunRequestV1) -> ResearchRunResponseV1:
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(
                capability="research.pipeline",
                intent="research_summarize",
            ),
        )
        result = await self.task_runner.run_task(task)
        return ResearchRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            graph_id=result.metadata.get("graph_id"),
            agent_ids=list(result.metadata.get("agent_ids") or []),
        )


def mount_research_routes(
    app: FastAPI,
    *,
    nexus_loop: NexusLoop,
    prefix: str = "/v1/research",
) -> ResearchRunService:
    service = ResearchRunService.from_nexus_loop(nexus_loop)
    router = APIRouter(prefix=prefix, tags=["research"])

    @router.post("/run", response_model=ResearchRunResponseV1)
    async def research_run(body: ResearchRunRequestV1) -> ResearchRunResponseV1:
        try:
            return await service.run_pipeline(body)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"research_pipeline_error: {exc.__class__.__name__}",
            ) from exc

    app.include_router(router)
    return service
