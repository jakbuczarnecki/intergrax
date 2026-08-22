# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from intergrax.applications._shared.harness_auth import (
    HarnessAuthenticatedPrincipal,
    require_harness_auth,
    resolve_harness_authenticated_principal,
)
from intergrax.applications._shared.harness_principal import (
    harness_principal_to_request_identity,
    reject_identity_assertion_conflicts,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from research_application.serving.schemas import ResearchRunRequestV1, ResearchRunResponseV1


@dataclass
class ResearchRunService:
    task_runner: UnifiedTaskRunner

    @classmethod
    def from_nexus_loop(cls, nexus_loop: NexusLoop) -> ResearchRunService:
        return cls(task_runner=UnifiedTaskRunner(nexus_loop))

    async def run_pipeline(
        self,
        body: ResearchRunRequestV1,
        *,
        authenticated_principal: HarnessAuthenticatedPrincipal | None = None,
    ) -> ResearchRunResponseV1:
        canonical: RequestIdentity | None = None
        if authenticated_principal is not None:
            canonical = harness_principal_to_request_identity(authenticated_principal)
            reject_identity_assertion_conflicts(
                canonical=canonical,
                asserted_tenant_id=body.tenant_id,
                asserted_user_id=body.user_id,
            )
            tenant_id = canonical.tenant_id
            user_id = canonical.user_id or body.user_id
        else:
            tenant_id = body.tenant_id
            user_id = body.user_id

        task_id, run_id = mint_intake_execution_identity()
        task = Task(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(
                capability="research.pipeline",
                intent="research_summarize",
            ),
            canonical_identity=canonical,
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
    async def research_run(
        body: ResearchRunRequestV1,
        _: None = Depends(require_harness_auth),
        authenticated_principal: HarnessAuthenticatedPrincipal | None = Depends(
            resolve_harness_authenticated_principal
        ),
    ) -> ResearchRunResponseV1:
        try:
            return await service.run_pipeline(
                body,
                authenticated_principal=authenticated_principal,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"research_pipeline_error: {exc.__class__.__name__}",
            ) from exc

    app.include_router(router)
    return service
