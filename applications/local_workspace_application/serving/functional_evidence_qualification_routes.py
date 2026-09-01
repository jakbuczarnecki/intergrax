# © Artur Czarnecki. All rights reserved.

"""Qualification-only HTTP surface for reading persisted functional evidence."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, Query, status

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidenceQueryRequest
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    FunctionalEvidenceRuntimeWiring,
    functional_evidence_wiring_extra_key,
)


def mount_functional_evidence_qualification_routes(
    app: FastAPI,
    *,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["local_workspace_qualification"])

    @router.get("/qualification/functional_evidence")
    def read_functional_evidence(
        tenant_id: str = Query(...),
        task_id: str = Query(...),
        run_id: str = Query(...),
    ) -> dict[str, object]:
        wiring = _resolve_functional_evidence_wiring(app)
        if wiring is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="functional_evidence_not_configured",
            )
        page = wiring.persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=tenant_id,
                task_id=validate_task_id(task_id),
                run_id=validate_run_id(run_id),
                page_size=100,
            ),
        )
        return {
            "tenant_id": tenant_id,
            "task_id": task_id,
            "run_id": run_id,
            "items": [item.model_dump(mode="json") for item in page.items],
            "item_count": len(page.items),
            "next_cursor": page.next_cursor,
        }

    app.include_router(router)


def _resolve_functional_evidence_wiring(app: FastAPI) -> FunctionalEvidenceRuntimeWiring | None:
    raw = getattr(app.state, functional_evidence_wiring_extra_key(), None)
    if isinstance(raw, FunctionalEvidenceRuntimeWiring):
        return raw
    return None


__all__ = ["mount_functional_evidence_qualification_routes"]
