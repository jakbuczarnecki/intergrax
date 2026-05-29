# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production HTTP routes for inbound Slack / Teams / lab interactions (§18, B.12)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.models import InteractionIntakeResponse


def create_interaction_intake_router(
    service: InteractionIntakeService,
    *,
    tags: list[str] | None = None,
    execute_default: bool = True,
) -> APIRouter:
    router = APIRouter(tags=tags or ["interactions"])

    @router.post("/intake", response_model=InteractionIntakeResponse)
    async def interaction_intake(
        request: Request,
        tenant: str = Query(default="default", description="Tenant id when payload omits team_id"),
        execute: bool = Query(
            default=execute_default,
            description="When true, run the normalized Task through NexusLoop",
        ),
    ) -> InteractionIntakeResponse:
        body = await request.body()
        headers = {key: value for key, value in request.headers.items()}
        content_type = request.headers.get("content-type", "")
        try:
            intake = await service.intake_http(
                headers=headers,
                body=body,
                content_type=content_type,
                tenant_id=tenant,
                execute=execute,
            )
        except ValueError as exc:
            message = str(exc)
            if "signature" in message.lower() or "Slack" in message or "Teams" in message:
                raise HTTPException(status_code=401, detail=message) from exc
            raise HTTPException(status_code=422, detail=message) from exc
        except TypeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        task = intake.task
        response = InteractionIntakeResponse(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            capability=task.context.capability,
            message=task.message,
            interaction_channel=InteractionIntakeService.interaction_channel(task),
            executed=intake.executed,
        )
        if intake.result is not None:
            response.state = intake.result.state.value
            response.answer = intake.result.answer
            response.run_id = intake.result.run_id
            response.resume_token = intake.result.summary.resume_token
            response.checkpoint_id = intake.result.summary.checkpoint_id
        return response

    return router
