# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle


def mount_local_workspace_readiness_routes(
    app: FastAPI,
    lifecycle: LocalWorkspaceHostLifecycle,
    *,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["local_workspace"])

    @router.get("/readiness")
    async def readiness() -> dict[str, object]:
        return {
            "ready": lifecycle.is_ready(),
            "accepts_new_work": lifecycle.accepts_new_work,
            "state": lifecycle.state.value,
            "detail": lifecycle.readiness_detail(),
            "components": [
                {
                    "name": component.name,
                    "enabled": component.enabled,
                    "required": component.required,
                    "healthy": component.healthy,
                    "detail": component.detail,
                }
                for component in lifecycle.component_health()
            ],
        }

    app.include_router(router)
