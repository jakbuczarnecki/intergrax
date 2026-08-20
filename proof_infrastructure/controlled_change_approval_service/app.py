# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from proof_infrastructure.controlled_change_approval_service.models import (
    ChangeApprovalResponseV1,
    RequestCountResponseV1,
)
from proof_infrastructure.controlled_change_approval_service.seed import (
    ORION_FIXTURE_CHANGE_ID,
    seed_orion_change_fixture,
)
from proof_infrastructure.controlled_change_approval_service.state import ChangeApprovalStore


def create_app(*, store: ChangeApprovalStore | None = None) -> FastAPI:
    change_store = store or ChangeApprovalStore()
    if change_store.get_change(ORION_FIXTURE_CHANGE_ID) is None:
        seed_orion_change_fixture(change_store)

    app = FastAPI(title="Controlled Change Approval Service", version="1")
    app.state.change_store = change_store

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/changes/{change_id}/approval", response_model=None)
    def read_change_approval(change_id: str) -> ChangeApprovalResponseV1:
        status = change_store.read_change(change_id)
        if status is None:
            raise HTTPException(status_code=404, detail="change_not_found")
        return status

    @app.get("/control/request-count", response_model=RequestCountResponseV1)
    def read_request_count() -> RequestCountResponseV1:
        return RequestCountResponseV1(
            read_request_count=change_store.read_request_count(),
        )

    @app.post("/control/request-count/reset", status_code=204)
    def reset_request_count() -> Response:
        change_store.reset_read_request_count()
        return Response(status_code=204)

    @app.exception_handler(HTTPException)
    def _http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "request_failed"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    return app
