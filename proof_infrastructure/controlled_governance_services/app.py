# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from proof_infrastructure.controlled_governance_services.models import (
    ChangeApprovalResponseV1,
    GovernanceApprovalResponseV1,
    RequestCountResponseV1,
    SecurityStatusResponseV1,
)
from proof_infrastructure.controlled_governance_services.seed import seed_orion_governance_fixture
from proof_infrastructure.controlled_governance_services.state import GovernanceServicesStore


def create_app(*, store: GovernanceServicesStore | None = None) -> FastAPI:
    governance_store = store or GovernanceServicesStore()
    seed_orion_governance_fixture(governance_store)

    app = FastAPI(title="Controlled Governance Services", version="1")
    app.state.governance_store = governance_store

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/projects/{project_id}/security-status", response_model=None)
    def read_security_status(project_id: str) -> SecurityStatusResponseV1:
        status = governance_store.get_security(project_id)
        if status is None:
            raise HTTPException(status_code=404, detail="project_not_found")
        return status

    @app.get("/changes/{change_id}/approval", response_model=None)
    def read_change_approval(change_id: str) -> ChangeApprovalResponseV1:
        status = governance_store.get_change(change_id)
        if status is None:
            raise HTTPException(status_code=404, detail="change_not_found")
        return status

    @app.get("/approvals/{subject_id}/status", response_model=None)
    def read_governance_approval(subject_id: str) -> GovernanceApprovalResponseV1:
        status = governance_store.get_governance(subject_id)
        if status is None:
            raise HTTPException(status_code=404, detail="subject_not_found")
        return status

    @app.get("/control/request-count", response_model=RequestCountResponseV1)
    def read_request_count() -> RequestCountResponseV1:
        security, change, governance = governance_store.read_counts()
        return RequestCountResponseV1(
            security_read_count=security,
            change_read_count=change,
            governance_read_count=governance,
        )

    @app.post("/control/request-count/reset", status_code=204)
    def reset_request_count() -> Response:
        governance_store.reset_read_counts()
        return Response(status_code=204)

    return app
