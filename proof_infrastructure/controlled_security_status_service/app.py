# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from proof_infrastructure.controlled_security_status_service.models import (
    RequestCountResponseV1,
    SecurityStatusReadBehaviorControlV1,
    SecurityStatusReadBehaviorV1,
    SecurityStatusResponseV1,
)
from proof_infrastructure.controlled_security_status_service.seed import seed_orion_security_fixture
from proof_infrastructure.controlled_security_status_service.state import SecurityStatusStore


def create_app(*, store: SecurityStatusStore | None = None) -> FastAPI:
    security_store = store or SecurityStatusStore()
    if security_store.get_security("ORION") is None:
        seed_orion_security_fixture(security_store)

    app = FastAPI(title="Controlled Security Status Service", version="1")
    app.state.security_store = security_store

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/projects/{project_id}/security-status", response_model=None)
    def read_security_status(project_id: str) -> SecurityStatusResponseV1:
        behavior = security_store.read_behavior()
        status = security_store.read_security(project_id)
        if behavior is SecurityStatusReadBehaviorV1.HTTP_503:
            raise HTTPException(status_code=503, detail="controlled_server_unavailable")
        if behavior is SecurityStatusReadBehaviorV1.MALFORMED_JSON:
            return Response(content="{not-valid-json", media_type="application/json")
        if status is None:
            raise HTTPException(status_code=404, detail="project_not_found")
        return status

    @app.get("/control/request-count", response_model=RequestCountResponseV1)
    def read_request_count() -> RequestCountResponseV1:
        return RequestCountResponseV1(
            read_request_count=security_store.read_request_count(),
        )

    @app.post("/control/request-count/reset", status_code=204)
    def reset_request_count() -> Response:
        security_store.reset_read_request_count()
        return Response(status_code=204)

    @app.put("/control/read-behavior", response_model=SecurityStatusReadBehaviorControlV1)
    def control_read_behavior(
        control: SecurityStatusReadBehaviorControlV1,
    ) -> SecurityStatusReadBehaviorControlV1:
        security_store.set_read_behavior(control.behavior)
        return control

    @app.exception_handler(HTTPException)
    def _http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "request_failed"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    return app
