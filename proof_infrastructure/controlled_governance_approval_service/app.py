# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from proof_infrastructure.controlled_governance_approval_service.models import (
    GovernanceApprovalResponseV1,
    GovernanceApprovalSeedControlV1,
    RequestCountResponseV1,
)
from proof_infrastructure.controlled_governance_approval_service.seed import (
    ORION_FIXTURE_SUBJECT_ID,
    seed_orion_governance_fixture,
)
from proof_infrastructure.controlled_governance_approval_service.mongo_state import (
    MongoGovernanceApprovalStore,
)
from proof_infrastructure.controlled_governance_approval_service.state import (
    GovernanceApprovalStore,
)


def create_app(
    *,
    store: GovernanceApprovalStore | MongoGovernanceApprovalStore | None = None,
) -> FastAPI:
    governance_store = store or GovernanceApprovalStore()
    if governance_store.get_governance(ORION_FIXTURE_SUBJECT_ID) is None:
        seed_orion_governance_fixture(governance_store)

    app = FastAPI(title="Controlled Governance Approval Service", version="1")
    app.state.governance_store = governance_store

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/approvals/{subject_id}/status", response_model=None)
    def read_governance_approval(subject_id: str) -> GovernanceApprovalResponseV1:
        status = governance_store.read_governance(subject_id)
        if status is None:
            raise HTTPException(status_code=404, detail="subject_not_found")
        return status

    @app.get("/control/request-count", response_model=RequestCountResponseV1)
    def read_request_count() -> RequestCountResponseV1:
        return RequestCountResponseV1(
            read_request_count=governance_store.read_request_count(),
        )

    @app.post("/control/request-count/reset", status_code=204)
    def reset_request_count() -> Response:
        governance_store.reset_read_request_count()
        return Response(status_code=204)

    @app.post("/control/seed-orion", response_model=GovernanceApprovalResponseV1)
    def control_seed_orion(
        control: GovernanceApprovalSeedControlV1 | None = None,
    ) -> GovernanceApprovalResponseV1:
        return seed_orion_governance_fixture(
            governance_store,
            valid_from=control.valid_from if control is not None else None,
            valid_until=control.valid_until if control is not None else None,
        )

    @app.get("/control/fixture/{subject_id}", response_model=GovernanceApprovalResponseV1)
    def control_read_fixture(subject_id: str) -> GovernanceApprovalResponseV1:
        status = governance_store.get_governance(subject_id)
        if status is None:
            raise HTTPException(status_code=404, detail="subject_not_found")
        return status

    @app.exception_handler(HTTPException)
    def _http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "request_failed"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    return app
