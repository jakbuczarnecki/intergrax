# © Artur Czarnecki. All rights reserved.

"""Trace Explorer HTTP routes for product ops surfaces (AUDIT-IDEAL-27.1)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from intergrax.debug.formatters import build_trace_payload
from intergrax.debug.store import resolve_trace_reader


def create_trace_explorer_router(
    *,
    enabled: bool = True,
    db_path: Path | None = None,
) -> APIRouter:
    """Expose read-only trace explorer endpoints beyond lab-only debug surface."""
    router = APIRouter(prefix="/ops/trace", tags=["trace-explorer"])

    @router.get("/runs/{run_id}")
    def get_run_trace(run_id: str, tenant_id: str = "default") -> dict[str, object]:
        if not enabled:
            raise HTTPException(status_code=404, detail="trace explorer disabled")
        reader = resolve_trace_reader(db_path=db_path)
        try:
            persisted = reader.read_run(run_id, tenant_id)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return build_trace_payload(persisted)

    @router.get("/health")
    def trace_explorer_health() -> dict[str, str]:
        if not enabled:
            raise HTTPException(status_code=404, detail="trace explorer disabled")
        return {"status": "ok", "surface": "trace-explorer"}

    return router
