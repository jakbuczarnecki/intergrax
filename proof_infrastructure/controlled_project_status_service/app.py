# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from proof_infrastructure.controlled_project_status_service.models import (
    ProjectStatusControlUpdateV1,
    ProjectStatusReadBehaviorControlV1,
    ProjectStatusReadBehaviorV1,
    ProjectStatusResponseV1,
    RequestCountResponseV1,
)
from proof_infrastructure.controlled_project_status_service.seed import seed_orion_fixture
from proof_infrastructure.controlled_project_status_service.mongo_state import (
    MongoProjectStatusStore,
)
from proof_infrastructure.controlled_project_status_service.state import ProjectStatusStore


def create_app(
    *,
    store: ProjectStatusStore | MongoProjectStatusStore | None = None,
) -> FastAPI:
    status_store = store or ProjectStatusStore()
    if not status_store.get_status("ORION"):
        seed_orion_fixture(status_store)

    app = FastAPI(title="Controlled Project Status Service", version="1")
    app.state.project_status_store = status_store

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/projects/{project_id}/status", response_model=None)
    def read_project_status(project_id: str) -> Response | ProjectStatusResponseV1:
        behavior = status_store.read_behavior()
        status = status_store.read_status(project_id)
        if behavior is ProjectStatusReadBehaviorV1.HTTP_500:
            raise HTTPException(status_code=500, detail="controlled_server_error")
        if behavior is ProjectStatusReadBehaviorV1.HTTP_503:
            raise HTTPException(status_code=503, detail="controlled_server_unavailable")
        if behavior is ProjectStatusReadBehaviorV1.MALFORMED_JSON:
            return Response(content="{not-valid-json", media_type="application/json")
        if behavior is ProjectStatusReadBehaviorV1.INVALID_SCHEMA:
            return JSONResponse(
                status_code=200,
                content={"unexpected_field": "contract_invalid"},
            )
        if status is None:
            raise HTTPException(status_code=404, detail="project_not_found")
        return status

    @app.put(
        "/control/projects/{project_id}/status",
        response_model=ProjectStatusResponseV1,
    )
    def control_project_status(
        project_id: str,
        update: ProjectStatusControlUpdateV1,
    ) -> ProjectStatusResponseV1:
        try:
            return status_store.update_status(project_id, update)
        except KeyError:
            raise HTTPException(status_code=404, detail="project_not_found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/control/request-count", response_model=RequestCountResponseV1)
    def read_request_count() -> RequestCountResponseV1:
        return RequestCountResponseV1(
            read_request_count=status_store.read_request_count(),
        )

    @app.post("/control/request-count/reset", status_code=204)
    def reset_request_count() -> Response:
        status_store.reset_read_request_count()
        return Response(status_code=204)

    @app.put("/control/read-behavior", response_model=ProjectStatusReadBehaviorControlV1)
    def control_read_behavior(
        control: ProjectStatusReadBehaviorControlV1,
    ) -> ProjectStatusReadBehaviorControlV1:
        status_store.set_read_behavior(control.behavior)
        return control

    @app.post("/control/seed-orion", response_model=ProjectStatusResponseV1)
    def control_seed_orion() -> ProjectStatusResponseV1:
        return seed_orion_fixture(status_store)

    @app.get("/control/fixture/{project_id}", response_model=ProjectStatusResponseV1)
    def control_read_fixture(project_id: str) -> ProjectStatusResponseV1:
        status = status_store.get_status(project_id)
        if status is None:
            raise HTTPException(status_code=404, detail="project_not_found")
        return status

    @app.exception_handler(HTTPException)
    def _http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "request_failed"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    return app
