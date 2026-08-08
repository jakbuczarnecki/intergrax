# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from local_workspace_application.host.readiness import LocalWorkspaceReadinessProvider


def mount_local_workspace_readiness_routes(
    app: FastAPI,
    readiness: LocalWorkspaceReadinessProvider,
    *,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["local_workspace"])

    @router.get("/liveness")
    async def liveness_endpoint() -> dict[str, bool]:
        """The HTTP event loop is alive when this endpoint can respond."""
        return {"alive": True}

    @router.get("/readiness")
    async def readiness_endpoint() -> dict[str, object]:
        snapshot = readiness.readiness_snapshot()
        return {
            "ready": snapshot.ready,
            "accepts_new_work": snapshot.accepts_new_work,
            "state": snapshot.state,
            "detail": snapshot.detail,
            "components": [
                {
                    "name": component.name,
                    "enabled": component.enabled,
                    "required": component.required,
                    "healthy": component.healthy,
                    "detail": component.detail,
                }
                for component in snapshot.components
            ],
        }

    app.include_router(router)
